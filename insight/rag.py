"""Core RAG engine orchestrating parsing, indexing, HybridRAG/AdaptiveRAG retrieval, LLM synthesis, and history."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from insight.config import RAGMode, Settings, normalize_rag_mode
from insight.llm import LazyLLM
from insight.parser import chunk_markdown, parse_file
from insight.storage import DocumentRecord, MessageRecord, ProjectRecord, SessionSummary, SQLiteStore
from insight.vectorstore import STOPWORDS, TOKEN_RE, LocalVectorStore, RetrievedChunk, _lexical_overlap

VALID_RAG_MODES: tuple[RAGMode, ...] = ("hybrid", "adaptive")
_UNSET: Any = object()


class QueryAnalyzer:
    """Analyze query characteristics to route AdaptiveRAG retrieval strategies."""

    temporal_keywords = (
        "today",
        "yesterday",
        "recent",
        "latest",
        "current",
        "now",
        "this year",
        "last month",
        "2024",
        "2025",
        "2026",
        "breaking",
        "news",
        "hari ini",
        "kemarin",
        "terbaru",
        "terkini",
        "sekarang",
        "tahun ini",
        "bulan lalu",
        "berita",
    )
    factual_keywords = (
        "what is",
        "define",
        "definition",
        "explain",
        "how does",
        "difference between",
        "compare",
        "overview",
        "summary",
        "apa itu",
        "apa yang dimaksud",
        "definisi",
        "jelaskan",
        "bagaimana",
        "perbedaan",
        "bandingkan",
        "ringkasan",
        "ringkas",
        "rangkum",
    )
    specific_keywords = (
        "specific",
        "exact",
        "precise",
        "detailed",
        "comprehensive",
        "in-depth",
        "thorough",
        "complete analysis",
        "spesifik",
        "rinci",
        "detail",
        "mendalam",
        "lengkap",
        "komprehensif",
        "analisis",
    )
    multi_source_keywords = (
        "compare",
        "versus",
        "vs",
        "different",
        "bandingkan",
        "perbedaan",
        "perbandingan",
        "dibanding",
    )

    @staticmethod
    def _contains_term(query_lower: str, term: str) -> bool:
        return bool(re.search(rf"\b{re.escape(term)}\b", query_lower))

    def analyze_query(self, query: str) -> dict[str, Any]:
        """Classify complexity, factuality, temporality, and multi-source intent."""
        query_lower = query.lower().strip()
        is_temporal = any(self._contains_term(query_lower, kw) for kw in self.temporal_keywords)
        is_factual = any(self._contains_term(query_lower, kw) for kw in self.factual_keywords)
        needs_multiple_sources = any(
            self._contains_term(query_lower, kw) for kw in self.multi_source_keywords
        )

        word_count = len(query.split())
        is_complex = (
            word_count > 10
            or any(self._contains_term(query_lower, kw) for kw in self.specific_keywords)
            or query.count("?") >= 2
        )

        if word_count <= 5 and is_factual and not is_complex and not needs_multiple_sources:
            complexity = "simple"
        elif word_count <= 10 and not is_complex:
            complexity = "medium"
        else:
            complexity = "complex"

        return {
            "complexity": complexity,
            "is_temporal": is_temporal,
            "is_factual": is_factual,
            "needs_multiple_sources": needs_multiple_sources,
            "word_count": word_count,
            "original_query": query,
        }


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[dict[str, Any]]
    session_id: str
    rag_mode: str = "hybrid"
    retrieval_strategy: str = "hybrid_search"
    query_analysis: dict[str, Any] | None = None


class InSightEngine:
    """Stateful workspace engine combining SQLite, Qdrant Local, lightweight parsers, and Dual RAG (HybridRAG / AdaptiveRAG)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.ensure_dirs()
        self.store = SQLiteStore(self.settings.sqlite_path)
        self.vectors = LocalVectorStore(
            qdrant_path=self.settings.qdrant_path,
            embed_model=self.settings.embed_model,
            use_reranker=self.settings.use_reranker,
            rerank_model=self.settings.rerank_model,
            strict_embedder=self.settings.strict_embedder,
        )
        self.llm = LazyLLM(self.settings)
        self.query_analyzer = QueryAnalyzer()

    @property
    def active_model_id(self) -> str:
        return self.llm.active_model_id

    def list_models(self) -> list[dict[str, Any]]:
        from insight.llm import list_available_models

        return list_available_models(self.settings)

    def switch_model(self, model_id: str) -> dict[str, Any]:
        result = self.llm.switch_model(model_id)
        self.settings = self.llm.settings
        return result

    @staticmethod
    def select_adaptive_strategy(analysis: dict[str, Any]) -> str:
        """Select optimal retrieval strategy from QueryAnalyzer output."""
        if (
            analysis["complexity"] == "simple"
            and analysis["is_factual"]
            and analysis["word_count"] <= 5
        ):
            return "direct_llm"
        if analysis["needs_multiple_sources"] or analysis["complexity"] == "complex":
            return "comprehensive"
        if analysis["complexity"] == "medium" or analysis["is_temporal"]:
            return "hybrid_search"
        return "simple_vector"

    @staticmethod
    def _grade_retrieval_relevance(query: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "not_relevant"
        max_lex = max(_lexical_overlap(query, item.text) for item in chunks)
        max_score = max(item.score for item in chunks)
        if max_lex > 0.0 or max_score >= 0.35:
            return "relevant"
        return "not_relevant"

    @staticmethod
    def _rewrite_query_for_retrieval(query: str) -> str:
        tokens = TOKEN_RE.findall(query.lower())
        keywords = [tok for tok in tokens if len(tok) >= 2 and tok not in STOPWORDS]
        deduped = list(dict.fromkeys(keywords))
        return " ".join(deduped) if deduped else query.strip()

    @staticmethod
    def _is_front_matter_query(query: str) -> bool:
        q_lower = query.lower()
        front_keywords = (
            "cover",
            "sampul",
            "judul",
            "penulis",
            "kelompok",
            "penyusun",
            "mahasiswa",
            "berkontribusi",
            "disusun",
            "anggota",
            "pengarang",
            "author",
            "authors",
            "title",
            "contributor",
        )
        return any(re.search(rf"\b{re.escape(kw)}\b", q_lower) for kw in front_keywords)

    @staticmethod
    def _contextualize_query(question: str, prior_messages: list[MessageRecord]) -> str:
        """Rewrite follow-up questions into self-contained retrieval queries using active topic context."""
        if not prior_messages:
            return question
        q_lower = question.lower().strip()
        words = q_lower.split()
        is_explicit_standalone = q_lower.startswith(
            ("apa itu ", "what is ", "define ", "jelaskan tentang ", "explain ")
        )
        if is_explicit_standalone:
            return question
        has_followup_cue = bool(
            re.search(
                r"\b(siapa saja|tersebut|sebelumnya|tadi|di atas|barusan|maksudnya|detailnya|"
                r"selain itu|apakah hanya|bagaimana dengan|lalu|terus|lanjut|lebih lanjut|"
                r"kenapa|mengapa|contohnya|alasannya|kelebihan|kekurangan|dampak|solusi|"
                r"poin|itu|ini|mereka|keduanya|that|this|those|these|it|them|above|"
                r"elaborate|more about|follow up)\b",
                q_lower,
            )
        )
        short_fragment = len(words) <= 5
        if not (has_followup_cue or short_fragment):
            return question

        prev_users = [
            m.content.strip()
            for m in prior_messages
            if m.role == "user" and m.content.strip()
        ]
        if not prev_users:
            return question

        last_user = prev_users[-1]
        anchor = next((u for u in reversed(prev_users) if len(u.split()) >= 4), last_user)
        if anchor.lower() == q_lower:
            return question

        context_parts = [anchor]
        if last_user.lower() not in {anchor.lower(), q_lower}:
            context_parts.append(last_user)
        return f"{' — '.join(context_parts)} — {question}"

    def _merge_front_matter(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        if not self._is_front_matter_query(query):
            return chunks
        front_chunks = self.vectors.get_front_matter_chunks(
            max_chunk_index=3,
            limit=2,
            doc_id=doc_id,
            doc_ids=doc_ids,
        )
        if not front_chunks:
            return chunks
        merged: list[RetrievedChunk] = []
        seen: set[tuple[str, int]] = set()
        for item in [*front_chunks, *chunks]:
            key = (item.doc_id, item.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max(self.settings.top_k_final, len(chunks)):
                break
        return merged

    def _retrieve_hybrid(
        self,
        query: str,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[RetrievedChunk], str]:
        chunks = self.vectors.search(
            query=query,
            top_k_candidates=self.settings.top_k_candidates,
            top_k_final=self.settings.top_k_final,
            strategy="hybrid_search",
            dense_weight=0.70,
            sparse_weight=0.30,
            doc_id=doc_id,
            doc_ids=doc_ids,
        )
        if self._grade_retrieval_relevance(query, chunks) == "not_relevant":
            rewritten = self._rewrite_query_for_retrieval(query)
            if rewritten and rewritten.lower() != query.lower():
                retry_chunks = self.vectors.search(
                    query=rewritten,
                    top_k_candidates=self.settings.top_k_candidates,
                    top_k_final=self.settings.top_k_final,
                    strategy="hybrid_search",
                    dense_weight=0.70,
                    sparse_weight=0.30,
                    doc_id=doc_id,
                    doc_ids=doc_ids,
                )
                if retry_chunks and (
                    not chunks
                    or max(c.score for c in retry_chunks) >= max(c.score for c in chunks)
                ):
                    return retry_chunks, "hybrid_search_rewritten"
        return chunks, "hybrid_search"

    def _retrieve_adaptive(
        self,
        query: str,
        doc_id: str | None = None,
        doc_ids: list[str] | None = None,
    ) -> tuple[list[RetrievedChunk], str, dict[str, Any]]:
        analysis = self.query_analyzer.analyze_query(query)
        strategy = self.select_adaptive_strategy(analysis)
        dense_weight = 0.8 if analysis["is_factual"] else 0.6
        sparse_weight = round(1.0 - dense_weight, 4)

        chunks = self.vectors.search(
            query=query,
            top_k_candidates=self.settings.top_k_candidates,
            top_k_final=self.settings.top_k_final,
            strategy=strategy,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            doc_id=doc_id,
            doc_ids=doc_ids,
        )
        if self._grade_retrieval_relevance(query, chunks) == "not_relevant" and strategy != "comprehensive":
            rewritten = self._rewrite_query_for_retrieval(query)
            escalated = self.vectors.search(
                query=rewritten,
                top_k_candidates=self.settings.top_k_candidates,
                top_k_final=self.settings.top_k_final,
                strategy="comprehensive",
                dense_weight=0.70,
                sparse_weight=0.30,
                doc_id=doc_id,
                doc_ids=doc_ids,
            )
            if escalated and (
                not chunks
                or max(c.score for c in escalated) >= max(c.score for c in chunks)
            ):
                return escalated, "comprehensive", analysis
        return chunks, strategy, analysis

    def ingest_file(
        self,
        file_path: Path,
        display_name: str | None = None,
        project_id: str | None = None,
    ) -> DocumentRecord:
        filename = display_name or file_path.name
        parsed = parse_file(file_path)
        if not parsed.markdown:
            raise ValueError(f"No readable content extracted from '{filename}'.")

        chunks = chunk_markdown(
            parsed.markdown,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError(f"Document '{filename}' produced zero text chunks.")

        existing_same_name = self.store.find_documents_by_filename(filename, project_id=project_id)

        doc_id = uuid.uuid4().hex[:12]
        self.vectors.upsert_chunks(doc_id=doc_id, filename=filename, chunks=chunks)
        try:
            record = self.store.add_document(
                doc_id=doc_id,
                filename=filename,
                file_ext=file_path.suffix.lower() or ".txt",
                chunk_count=len(chunks),
                char_count=len(parsed.markdown),
                parser_used=parsed.parser_used,
                project_id=project_id,
            )
        except Exception:
            self.vectors.delete_document(doc_id)
            raise

        for old_doc in existing_same_name:
            if old_doc.doc_id != doc_id:
                self.vectors.delete_document(old_doc.doc_id)
                self.store.delete_document(old_doc.doc_id)

        return record

    def list_documents(self) -> list[DocumentRecord]:
        return self.store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        record = self.store.get_document(doc_id)
        if record is None:
            return False
        self.vectors.delete_document(doc_id)
        self.store.delete_document(doc_id)
        return True

    def rename_document(self, doc_id: str, new_name: str) -> DocumentRecord | None:
        cleaned = Path(new_name.strip()).name
        if not cleaned or cleaned in {".", ".."}:
            raise ValueError("Document name cannot be empty.")
        record = self.store.get_document(doc_id)
        if record is None:
            return None
        if cleaned == record.filename:
            return record
        updated = self.store.rename_document(doc_id, cleaned)
        if updated is None:
            return None
        self.vectors.rename_document(doc_id, cleaned)
        return updated

    def create_project(self, name: str) -> ProjectRecord:
        project_id = f"proj-{uuid.uuid4().hex[:10]}"
        return self.store.create_project(project_id=project_id, name=name)

    def list_projects(self) -> list[ProjectRecord]:
        return self.store.list_projects()

    def rename_project(self, project_id: str, name: str) -> ProjectRecord | None:
        return self.store.rename_project(project_id=project_id, name=name)

    def delete_project(self, project_id: str) -> bool:
        return self.store.delete_project(project_id=project_id)

    def set_document_project(self, doc_id: str, project_id: str | None) -> DocumentRecord | None:
        return self.store.set_document_project(doc_id=doc_id, project_id=project_id)

    def _prepare_ask(
        self,
        question: str,
        session_id: str = "default",
        rag_mode: str | None = None,
        doc_id: Any = _UNSET,
        project_id: Any = _UNSET,
    ) -> tuple[
        str,
        RAGMode,
        str,
        dict[str, Any] | None,
        list[dict[str, str]],
        list[RetrievedChunk],
        list[dict[str, Any]],
        str | None,
    ]:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty.")

        selected_mode = normalize_rag_mode(
            rag_mode if rag_mode is not None else self.settings.rag_mode,
            default=self.settings.rag_mode,
            strict=True,
        )
        query_analysis: dict[str, Any] | None = (
            self.query_analyzer.analyze_query(cleaned) if selected_mode == "adaptive" else None
        )

        prior_messages = self.store.list_messages(session_id=session_id)
        chat_history = [
            {"role": m.role, "content": m.content}
            for m in prior_messages[-8:]
            if m.role in {"user", "assistant"}
        ]
        retrieval_query = self._contextualize_query(cleaned, prior_messages)

        docs = self.store.list_documents()
        if not docs:
            empty_answer = (
                "Belum ada dokumen yang diunggah untuk menjawab pertanyaan tersebut. "
                "Silakan klik '+' atau 'Unggah Dokumen' di panel kiri untuk menambahkan dokumen terlebih dahulu."
            )
            self.store.add_message(session_id=session_id, role="user", content=cleaned)
            self.store.add_message(
                session_id=session_id,
                role="assistant",
                content=empty_answer,
                sources=[],
            )
            return (
                cleaned,
                selected_mode,
                "none",
                query_analysis,
                chat_history,
                [],
                [],
                empty_answer,
            )

        if doc_id is not _UNSET:
            clean_doc = doc_id.strip() if isinstance(doc_id, str) else None
            self.store.set_session_doc(session_id=session_id, doc_id=clean_doc or None)
            if clean_doc and project_id is _UNSET:
                self.store.set_session_project(session_id=session_id, project_id=None)
        if project_id is not _UNSET:
            clean_proj = project_id.strip() if isinstance(project_id, str) else None
            self.store.set_session_project(session_id=session_id, project_id=clean_proj or None)
            if clean_proj and doc_id is _UNSET:
                self.store.set_session_doc(session_id=session_id, doc_id=None)
        effective_doc_id = self.store.get_session_doc(session_id)
        effective_project_id = self.store.get_session_project(session_id)
        effective_doc_ids = (
            self.store.list_project_doc_ids(effective_project_id)
            if effective_project_id and not effective_doc_id
            else None
        )

        if selected_mode == "adaptive":
            retrieved, retrieval_strategy, search_analysis = self._retrieve_adaptive(
                retrieval_query,
                doc_id=effective_doc_id,
                doc_ids=effective_doc_ids,
            )
            if query_analysis is None:
                query_analysis = search_analysis
        else:
            retrieved, retrieval_strategy = self._retrieve_hybrid(
                retrieval_query,
                doc_id=effective_doc_id,
                doc_ids=effective_doc_ids,
            )

        retrieved = self._merge_front_matter(
            retrieval_query,
            retrieved,
            doc_id=effective_doc_id,
            doc_ids=effective_doc_ids,
        )
        sources = [
            {
                "doc_id": item.doc_id,
                "filename": item.filename,
                "section": item.section,
                "page_label": item.page_label,
                "chunk_index": item.chunk_index,
                "score": round(item.score, 4),
                "excerpt": item.text[:220],
            }
            for item in retrieved
        ]
        return (
            cleaned,
            selected_mode,
            retrieval_strategy,
            query_analysis,
            chat_history,
            retrieved,
            sources,
            None,
        )

    def ask(
        self,
        question: str,
        session_id: str = "default",
        rag_mode: str | None = None,
        doc_id: Any = _UNSET,
        project_id: Any = _UNSET,
        sampling_params: dict[str, Any] | None = None,
    ) -> ChatResult:
        (
            cleaned,
            selected_mode,
            retrieval_strategy,
            query_analysis,
            chat_history,
            retrieved,
            sources,
            empty_answer,
        ) = self._prepare_ask(
            question,
            session_id=session_id,
            rag_mode=rag_mode,
            doc_id=doc_id,
            project_id=project_id,
        )

        if empty_answer is not None:
            return ChatResult(
                answer=empty_answer,
                sources=[],
                session_id=session_id,
                rag_mode=selected_mode,
                retrieval_strategy="none",
                query_analysis=query_analysis,
            )

        if sampling_params:
            try:
                answer = self.llm.generate(
                    question=cleaned,
                    contexts=retrieved,
                    history=chat_history,
                    sampling_params=sampling_params,
                )
            except TypeError:
                answer = self.llm.generate(question=cleaned, contexts=retrieved, history=chat_history)
        else:
            answer = self.llm.generate(question=cleaned, contexts=retrieved, history=chat_history)
        self.store.add_message(session_id=session_id, role="user", content=cleaned)
        self.store.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        return ChatResult(
            answer=answer,
            sources=sources,
            session_id=session_id,
            rag_mode=selected_mode,
            retrieval_strategy=retrieval_strategy,
            query_analysis=query_analysis,
        )

    def ask_stream(
        self,
        question: str,
        session_id: str = "default",
        rag_mode: str | None = None,
        doc_id: Any = _UNSET,
        project_id: Any = _UNSET,
        sampling_params: dict[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        (
            cleaned,
            selected_mode,
            retrieval_strategy,
            query_analysis,
            chat_history,
            retrieved,
            sources,
            empty_answer,
        ) = self._prepare_ask(
            question,
            session_id=session_id,
            rag_mode=rag_mode,
            doc_id=doc_id,
            project_id=project_id,
        )

        yield {
            "type": "meta",
            "session_id": session_id,
            "rag_mode": selected_mode,
            "retrieval_strategy": retrieval_strategy,
            "query_analysis": query_analysis,
            "sources": sources,
        }

        if empty_answer is not None:
            yield {"type": "token", "delta": empty_answer}
            yield {"type": "done", "answer": empty_answer}
            return

        parts: list[str] = []
        if sampling_params:
            try:
                stream_iter = self.llm.generate_stream(
                    question=cleaned,
                    contexts=retrieved,
                    history=chat_history,
                    sampling_params=sampling_params,
                )
            except TypeError:
                stream_iter = self.llm.generate_stream(
                    question=cleaned,
                    contexts=retrieved,
                    history=chat_history,
                )
        else:
            stream_iter = self.llm.generate_stream(
                question=cleaned,
                contexts=retrieved,
                history=chat_history,
            )
        for delta in stream_iter:
            if delta:
                parts.append(delta)
                yield {"type": "token", "delta": delta}
        answer = "".join(parts).strip()
        self.store.add_message(session_id=session_id, role="user", content=cleaned)
        self.store.add_message(
            session_id=session_id,
            role="assistant",
            content=answer,
            sources=sources,
        )
        yield {"type": "done", "answer": answer}

    def get_history(self, session_id: str = "default") -> list[MessageRecord]:
        return self.store.list_messages(session_id=session_id)

    def clear_history(self, session_id: str = "default") -> int:
        return self.store.clear_messages(session_id=session_id)

    def clear_all_history(self) -> int:
        """Delete all conversation history across all sessions."""
        return self.store.clear_all_messages()

    def list_sessions(self) -> list[SessionSummary]:
        return self.store.list_sessions()

    def rename_session(self, session_id: str, title: str) -> bool:
        return self.store.rename_session(session_id=session_id, title=title)

    def close(self) -> None:
        self.vectors.close()
