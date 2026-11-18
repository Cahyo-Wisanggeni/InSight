"""Local Qdrant vector store with hybrid scoring and optional cross-encoder reranking."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from insight.parser import TextChunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    filename: str
    chunk_index: int
    section: str
    text: str
    score: float


class Embedder(Protocol):
    dimension: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Zero-dependency deterministic fallback embedder for offline tests and smoke runs."""

    dimension = 256
    name = "hashing-fallback-256d"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            tokens = TOKEN_RE.findall(text.lower())
            for tok in tokens:
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class FastEmbedProvider:
    """Dense embeddings via fastembed (CPU-friendly ONNX runtime)."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
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


def build_embedder(model_name: str) -> Embedder:
    try:
        return FastEmbedProvider(model_name)
    except Exception:
        pass
    try:
        return SentenceTransformerProvider(model_name)
    except Exception:
        return HashingEmbedder()


def _lexical_overlap(query: str, text: str) -> float:
    q_tokens = set(TOKEN_RE.findall(query.lower()))
    if not q_tokens:
        return 0.0
    t_tokens = set(TOKEN_RE.findall(text.lower()))
    return len(q_tokens & t_tokens) / len(q_tokens)


class LocalVectorStore:
    """Embedded Qdrant vector store persisted directly to a local directory."""

    COLLECTION_NAME = "insight_chunks"

    def __init__(
        self,
        qdrant_path: Path,
        embed_model: str = "BAAI/bge-small-en-v1.5",
        use_reranker: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._models = models
        qdrant_path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(qdrant_path))
        self.embedder = build_embedder(embed_model)
        self.use_reranker = use_reranker
        self.rerank_model = rerank_model
        self._reranker = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.COLLECTION_NAME):
            info = self.client.get_collection(self.COLLECTION_NAME)
            existing_size = getattr(info.config.params.vectors, "size", None)
            if existing_size == self.embedder.dimension:
                return
            self.client.delete_collection(self.COLLECTION_NAME)

        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=self._models.VectorParams(
                size=self.embedder.dimension,
                distance=self._models.Distance.COSINE,
            ),
        )

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
                        "text": chunk.text,
                    },
                )
            )
        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)
        return len(points)

    def delete_document(self, doc_id: str) -> None:
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

    def search(
        self,
        query: str,
        top_k_candidates: int = 12,
        top_k_final: int = 4,
    ) -> list[RetrievedChunk]:
        query_vec = self.embedder.embed([query])[0]
        response = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=query_vec,
            limit=top_k_candidates,
            with_payload=True,
        )
        points = response.points
        if not points:
            return []

        scored: list[RetrievedChunk] = []
        for pt in points:
            payload = pt.payload or {}
            text = str(payload.get("text", ""))
            hybrid_score = (0.75 * float(pt.score)) + (0.25 * _lexical_overlap(query, text))
            scored.append(
                RetrievedChunk(
                    doc_id=str(payload.get("doc_id", "")),
                    filename=str(payload.get("filename", "")),
                    chunk_index=int(payload.get("chunk_index", 0)),
                    section=str(payload.get("section", "General")),
                    text=text,
                    score=hybrid_score,
                )
            )

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
                )
                for item, r_score in zip(scored, rerank_scores)
            ]

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k_final]
