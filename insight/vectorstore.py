"""Local Qdrant vector store with hybrid scoring and optional cross-encoder reranking."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from insight.parser import TextChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
TOC_DOTS_RE = re.compile(r"(?:\.{4,}|(?:\.\s){4,})")
TOC_LINE_RE = re.compile(r"(?:\.{3,}|(?:\.\s){3,})\s*(?:[ivxlcdm]+|\d+)\s*$", re.IGNORECASE)
STOPWORDS = {
    # Indonesian common stopwords
    "apa",
    "apakah",
    "bagaimana",
    "berapa",
    "bisa",
    "dari",
    "dan",
    "dengan",
    "di",
    "ini",
    "itu",
    "juga",
    "ke",
    "karena",
    "mana",
    "pada",
    "saja",
    "sebagai",
    "siapa",
    "tidak",
    "untuk",
    "yang",
    "atau",
    "dalam",
    "adalah",
    "oleh",
    "akan",
    "sudah",
    "telah",
    "tentang",
    # English common stopwords
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    filename: str
    chunk_index: int
    section: str
    text: str
    score: float
    page_label: str = ""


class Embedder(Protocol):
    dimension: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Zero-dependency deterministic fallback embedder for offline tests and smoke runs."""

    dimension = 384
    name = "local-hybrid-384d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        from collections import Counter

        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            normalized = text.lower().strip()
            tokens = TOKEN_RE.findall(normalized)
            features: list[str] = list(tokens)
            features.extend(f"{tokens[i]}_{tokens[i + 1]}" for i in range(len(tokens) - 1))
            compact = " ".join(tokens)
            if len(compact) >= 4:
                features.extend(compact[i : i + 4] for i in range(len(compact) - 3))

            counts = Counter(features)
            for feat, count in counts.items():
                weight = 1.0 + math.log(count)
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[idx] += sign * weight
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FastEmbedProvider:
    """Dense embeddings via fastembed (CPU-friendly ONNX runtime)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        cache_dir = os.getenv("FASTEMBED_CACHE_PATH")
        kwargs = {"cache_dir": cache_dir} if cache_dir else {}
        self._model = TextEmbedding(model_name=model_name, **kwargs)
        probe = list(self._model.embed(["probe"]))[0]
        self.dimension = len(probe)
        self.name = f"fastembed:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]


class SentenceTransformerProvider:
    """Dense embeddings via sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{model_name}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        arr = self._model.encode(texts, normalize_embeddings=True)
        return [row.tolist() for row in arr]


def build_embedder(model_name: str, strict: bool = False) -> Embedder:
    try:
        return FastEmbedProvider(model_name)
    except Exception:
        pass
    try:
        return SentenceTransformerProvider(model_name)
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"Strict embedder mode is enabled (INSIGHT_STRICT_EMBEDDER=true), "
                f"but neural embedder '{model_name}' could not be loaded. "
                "Install 'fastembed' or 'sentence-transformers' to continue."
            ) from exc
        return HashingEmbedder()


def _toc_noise_ratio(text: str) -> float:
    """Return fraction of lines in text that look like table-of-contents dot leaders."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    toc_lines = sum(1 for line in lines if TOC_LINE_RE.search(line) or TOC_DOTS_RE.search(line))
    return toc_lines / len(lines)


def _strip_toc_noise(text: str) -> str:
    """Remove dot-leader sequences so TOC formatting does not distort lexical scoring."""
    return TOC_DOTS_RE.sub(" ", text)


def _lexical_overlap(query: str, text: str) -> float:
    raw_q_tokens = set(TOKEN_RE.findall(query.lower()))
    if not raw_q_tokens:
        return 0.0
    filtered_q_tokens = {tok for tok in raw_q_tokens if len(tok) >= 2 and tok not in STOPWORDS}
    q_tokens = filtered_q_tokens or raw_q_tokens
    t_tokens = set(TOKEN_RE.findall(_strip_toc_noise(text).lower()))
    return len(q_tokens & t_tokens) / len(q_tokens)


def _tokenize_bm25(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    filtered = [tok for tok in tokens if len(tok) >= 2 and tok not in STOPWORDS]
    return filtered or tokens


def _normalize_scores(scores: list[float], *, preserve_positive_unit: bool = True) -> list[float]:
    """Normalize scores to [0.0, 1.0], preserving positive scores when min == max."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        if hi <= 0.0:
            return [0.0] * len(scores)
        if preserve_positive_unit:
            return [min(1.0, max(0.0, s)) for s in scores]
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _bm25_scores(
    query: str,
    texts: list[str],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute lightweight BM25Okapi relevance scores across candidate texts."""
    if not texts:
        return []
    q_tokens = _tokenize_bm25(query)
    if not q_tokens:
        return [0.0] * len(texts)

    from collections import Counter

    doc_tokens_list = [_tokenize_bm25(_strip_toc_noise(t)) for t in texts]
    doc_tfs = [Counter(tokens) for tokens in doc_tokens_list]
    doc_lens = [len(tokens) for tokens in doc_tokens_list]
    avg_dl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 1.0
    if avg_dl <= 0.0:
        avg_dl = 1.0

    n_docs = len(texts)
    unique_q = list(dict.fromkeys(q_tokens))
    scores = [0.0] * n_docs

    for term in unique_q:
        df = sum(1 for tf in doc_tfs if tf.get(term, 0) > 0)
        if df == 0:
            continue
        # Use non-negative Lucene/BM25+ style IDF so single-document corpora stay > 0.
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for idx, tf in enumerate(doc_tfs):
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            denom = freq + k1 * (1.0 - b + b * (doc_lens[idx] / avg_dl))
            scores[idx] += idf * ((freq * (k1 + 1.0)) / max(denom, 1e-9))

    return scores


class LocalVectorStore:
    """Embedded Qdrant vector store persisted directly to a local directory."""

    COLLECTION_NAME = "insight_chunks"

    def __init__(
        self,
        qdrant_path: Path,
        embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        use_reranker: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        strict_embedder: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._models = models
        self._lock = threading.RLock()
        self.embedder = build_embedder(embed_model)
        if strict_embedder and self.is_fallback_embedder:
            raise RuntimeError(
                f"Strict embedder mode is enabled (INSIGHT_STRICT_EMBEDDER=true), "
                f"but neural embedder '{embed_model}' could not be loaded "
                "(only HashingEmbedder fallback is available). "
                "Install 'fastembed' or 'sentence-transformers' to continue."
            )
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self._meta_path = qdrant_path / ".insight_embedder.json"
        self.migrated_points = 0
        self.client = QdrantClient(path=str(qdrant_path))
        self.use_reranker = use_reranker
        self.rerank_model = rerank_model
        self._reranker = None
        self._corpus_cache: list[tuple[Any, dict[str, Any]]] | None = None
        self._ensure_collection()

    @property
    def is_fallback_embedder(self) -> bool:
        return isinstance(self.embedder, HashingEmbedder)

    @property
    def embedder_warning(self) -> str | None:
        if not self.is_fallback_embedder:
            return None
        return (
            "Mode fallback HashingEmbedder aktif (fastembed/sentence-transformers tidak tersedia). "
            "Kualitas pencarian semantik berkurang; install fastembed untuk embedding neural."
        )

    def _read_recorded_embedder(self) -> str | None:
        if not self._meta_path.is_file():
            return None
        try:
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            name = data.get("embedder")
            return str(name) if name else None
        except Exception:
            return None

    def _write_recorded_embedder(self) -> None:
        self._meta_path.write_text(
            json.dumps({"embedder": self.embedder.name, "dimension": self.embedder.dimension}, indent=2),
            encoding="utf-8",
        )

    def _scroll_all_points(self) -> list[tuple[Any, dict[str, Any]]]:
        items: list[tuple[Any, dict[str, Any]]] = []
        offset = None
        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                limit=128,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for rec in records:
                items.append((rec.id, dict(rec.payload or {})))
            if next_offset is None or not records:
                break
            offset = next_offset
        return items

    def _reembed_points(self, stored: list[tuple[Any, dict[str, Any]]]) -> int:
        if not stored:
            return 0
        total = 0
        batch_size = 64
        for start in range(0, len(stored), batch_size):
            batch = stored[start : start + batch_size]
            texts = [str(payload.get("text", "")) for _, payload in batch]
            vectors = self.embedder.embed(texts)
            points = [
                self._models.PointStruct(id=pt_id, vector=vec, payload=payload)
                for (pt_id, payload), vec in zip(batch, vectors)
            ]
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
            total += len(points)
        return total

    def _ensure_collection(self) -> None:
        with self._lock:
            saved_points: list[tuple[Any, dict[str, Any]]] = []
            if self.client.collection_exists(self.COLLECTION_NAME):
                info = self.client.get_collection(self.COLLECTION_NAME)
                existing_size = getattr(info.config.params.vectors, "size", None)
                point_count = int(getattr(info, "points_count", 0) or 0)
                recorded_name = self._read_recorded_embedder()
                same_size = existing_size == self.embedder.dimension
                same_embedder = (recorded_name == self.embedder.name) or (
                    recorded_name is None and point_count == 0
                )
                if same_size and same_embedder:
                    self._write_recorded_embedder()
                    return
                if point_count > 0:
                    saved_points = self._scroll_all_points()
                self.client.delete_collection(self.COLLECTION_NAME)

            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=self._models.VectorParams(
                    size=self.embedder.dimension,
                    distance=self._models.Distance.COSINE,
                ),
            )
            if saved_points:
                self.migrated_points = self._reembed_points(saved_points)
            self._corpus_cache = None
            self._write_recorded_embedder()

    def _get_corpus_points(self) -> list[tuple[Any, dict[str, Any]]]:
        with self._lock:
            if self._corpus_cache is None:
                self._corpus_cache = self._scroll_all_points()
            return list(self._corpus_cache)

    def _get_reranker(self):
        if not self.use_reranker:
            return None
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.rerank_model)
            except Exception:
                self._reranker = False
        return None if self._reranker is False else self._reranker

    def upsert_chunks(self, doc_id: str, filename: str, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        points = []
        for chunk, vec in zip(chunks, vectors):
            point_seed = f"{doc_id}:{chunk.chunk_index}".encode("utf-8")
            point_id = int.from_bytes(hashlib.blake2b(point_seed, digest_size=8).digest(), "little")
            points.append(
                self._models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload={
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": chunk.chunk_index,
                        "section": chunk.section,
                        "page_label": chunk.page_label,
                        "text": chunk.text,
                    },
                )
            )
        with self._lock:
            self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
            self._corpus_cache = None
        return len(points)

    def delete_document(self, doc_id: str) -> None:
        with self._lock:
            self.client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=self._models.FilterSelector(
                    filter=self._models.Filter(
                        must=[
                            self._models.FieldCondition(
                                key="doc_id",
                                match=self._models.MatchValue(value=doc_id),
                            )
                        ]
                    )
                ),
            )
            self._corpus_cache = None

    def _dedupe_chunks(self, items: list[RetrievedChunk], limit: int) -> list[RetrievedChunk]:
        deduped: list[RetrievedChunk] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for item in items:
            key = (item.filename, item.section, item.text.strip())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
            if len(deduped) >= limit:
                break
        return deduped

    def search(
        self,
        query: str,
        top_k_candidates: int = 12,
        top_k_final: int = 4,
        strategy: str = "hybrid_search",
        dense_weight: float = 0.75,
        sparse_weight: float = 0.25,
    ) -> list[RetrievedChunk]:
        strategy_norm = (strategy or "hybrid_search").strip().lower()
        effective_limit = (
            max(top_k_candidates, top_k_final * 3)
            if strategy_norm == "comprehensive"
            else max(top_k_candidates, top_k_final)
        )
        effective_top_k = max(
            1,
            min(top_k_final, 2) if strategy_norm == "direct_llm" else top_k_final,
        )

        if strategy_norm == "simple_vector":
            dw, sw = 0.90, 0.10
        elif strategy_norm == "direct_llm":
            dw, sw = 0.80, 0.20
        else:
            total_w = dense_weight + sparse_weight
            if total_w <= 0:
                dw, sw = 0.75, 0.25
            else:
                dw, sw = dense_weight / total_w, sparse_weight / total_w

        query_vec = self.embedder.embed([query])[0]
        with self._lock:
            response = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=query_vec,
                limit=effective_limit,
                with_payload=True,
            )
        dense_points = response.points
        corpus = self._get_corpus_points()
        if not dense_points and not corpus:
            return []

        dense_by_id: dict[Any, tuple[dict[str, Any], float]] = {
            pt.id: (dict(pt.payload or {}), min(1.0, max(0.0, float(pt.score))))
            for pt in dense_points
        }

        sparse_by_id: dict[Any, float] = {}
        corpus_payload_by_id: dict[Any, dict[str, Any]] = {}
        top_sparse_ids: list[Any] = []
        if corpus:
            corpus_texts = [str(payload.get("text", "")) for _, payload in corpus]
            bm25_norm = _normalize_scores(_bm25_scores(query, corpus_texts), preserve_positive_unit=False)
            sparse_candidates: list[tuple[Any, float]] = []
            for (pt_id, payload), b_score, text in zip(corpus, bm25_norm, corpus_texts):
                corpus_payload_by_id[pt_id] = payload
                l_score = _lexical_overlap(query, text)
                s_score = (0.6 * b_score) + (0.4 * l_score)
                if s_score > 0.0:
                    sparse_by_id[pt_id] = s_score
                    sparse_candidates.append((pt_id, s_score))
            sparse_candidates.sort(key=lambda item: item[1], reverse=True)
            top_sparse_ids = [pt_id for pt_id, _ in sparse_candidates[:effective_limit]]

        missing_dense_ids = [pt_id for pt_id in top_sparse_ids if pt_id not in dense_by_id]
        if missing_dense_ids:
            with self._lock:
                extra = self.client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    query=query_vec,
                    query_filter=self._models.Filter(
                        must=[self._models.HasIdCondition(has_id=missing_dense_ids)]
                    ),
                    limit=len(missing_dense_ids),
                    with_payload=True,
                )
            for pt in extra.points:
                dense_by_id[pt.id] = (
                    dict(pt.payload or corpus_payload_by_id.get(pt.id, {})),
                    min(1.0, max(0.0, float(pt.score))),
                )

        candidate_ids = list(dict.fromkeys([*dense_by_id.keys(), *top_sparse_ids]))
        if not candidate_ids:
            return []

        payloads: list[dict[str, Any]] = []
        texts: list[str] = []
        dense_scores: list[float] = []
        sparse_scores: list[float] = []
        for pt_id in candidate_ids:
            payload, d_score = dense_by_id.get(pt_id, (corpus_payload_by_id.get(pt_id, {}), 0.0))
            payloads.append(payload)
            texts.append(str(payload.get("text", "")))
            dense_scores.append(d_score)
            sparse_scores.append(sparse_by_id.get(pt_id, 0.0))

        def _make_chunk(payload: dict, text: str, score: float) -> RetrievedChunk:
            return RetrievedChunk(
                doc_id=str(payload.get("doc_id", "")),
                filename=str(payload.get("filename", "")),
                chunk_index=int(payload.get("chunk_index", 0)),
                section=str(payload.get("section", "General")),
                text=text,
                score=score,
                page_label=str(payload.get("page_label", "")),
            )

        scored: list[RetrievedChunk] = []
        for payload, text, d_score, s_score in zip(payloads, texts, dense_scores, sparse_scores):
            raw_score = (dw * d_score) + (sw * s_score)
            if _toc_noise_ratio(text) >= 0.30:
                raw_score *= 0.45
            scored.append(_make_chunk(payload, text, raw_score))

        reranker = self._get_reranker()
        if reranker is not None:
            pairs = [(query, item.text) for item in scored]
            rerank_scores = reranker.predict(pairs)
            scored = [
                RetrievedChunk(
                    doc_id=item.doc_id,
                    filename=item.filename,
                    chunk_index=item.chunk_index,
                    section=item.section,
                    text=item.text,
                    score=float(r_score),
                    page_label=item.page_label,
                )
                for item, r_score in zip(scored, rerank_scores)
            ]

        scored.sort(key=lambda item: item.score, reverse=True)
        if strategy_norm == "comprehensive" and reranker is None:
            dense_only = sorted(
                [
                    _make_chunk(payload, text, d_score)
                    for payload, text, d_score in zip(payloads, texts, dense_scores)
                ],
                key=lambda item: item.score,
                reverse=True,
            )
            half = max(1, effective_top_k // 2)
            combined = scored[:half] + dense_only[: max(1, effective_top_k - half)] + scored[half:]
            return self._dedupe_chunks(combined, effective_top_k)

        return self._dedupe_chunks(scored, effective_top_k)

    def get_front_matter_chunks(self, max_chunk_index: int = 3, limit: int = 2) -> list[RetrievedChunk]:
        """Return earliest document chunks (cover/title/authors), prioritizing non-TOC content."""
        with self._lock:
            records, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=self._models.Filter(
                    must=[
                        self._models.FieldCondition(
                            key="chunk_index",
                            range=self._models.Range(lte=max_chunk_index),
                        )
                    ]
                ),
                limit=max(12, limit * 4),
                with_payload=True,
                with_vectors=False,
            )
        candidates: list[tuple[int, str, int, RetrievedChunk]] = []
        for rec in records:
            payload = rec.payload or {}
            text = str(payload.get("text", ""))
            if not text.strip():
                continue
            toc_flag = 1 if _toc_noise_ratio(text) >= 0.45 else 0
            c_idx = int(payload.get("chunk_index", 0))
            fname = str(payload.get("filename", ""))
            chunk = RetrievedChunk(
                doc_id=str(payload.get("doc_id", "")),
                filename=fname,
                chunk_index=c_idx,
                section=str(payload.get("section", "General")),
                text=text,
                score=0.85 if toc_flag == 0 else 0.30,
                page_label=str(payload.get("page_label", "")),
            )
            candidates.append((toc_flag, fname, c_idx, chunk))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in candidates[:limit]]

    def close(self) -> None:
        with self._lock:
            try:
                self.client.close()
            except Exception:
                pass
