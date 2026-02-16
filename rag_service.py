#!/usr/bin/env python3
"""
RAG Service

Retrieval-Augmented Generation using in-memory documents.
Documents are loaded from a folder (e.g. data/) via text_loader. No database.
"""

from typing import Dict, Any, Tuple, List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from text_loader import load_path, load_sources

# Chunk size and overlap for splitting long documents
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100


class RAGService:
    """
    RAG service that loads documents from a path, builds embeddings,
    and retrieves relevant context for questions.
    """

    def __init__(
        self,
        data_path: str = "data",
        sources: Optional[List[Any]] = None,
    ):
        """
        data_path: folder or file path to load (used if sources is None).
        sources: optional list of path strings and/or API dicts {"url": "...", "method": "GET", ...}.
        """
        self.data_path = data_path
        self.sources = sources
        self.embedding_model = None
        self.embeddings_cache: Dict[str, Any] = {}
        self._init_model()
        self._load_and_embed()

    def _init_model(self) -> None:
        """Load the sentence transformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[OK] RAG embedding model ready")
        except Exception as e:
            print(f"[FAIL] Embedding model: {e}")
            self.embedding_model = None

    def _load_and_embed(self) -> None:
        """Load documents from data_path and compute embeddings."""
        if not self.embedding_model:
            return

        try:
            if self.sources is not None:
                documents = load_sources(self.sources)
            else:
                documents = load_path(self.data_path)
            if not documents:
                print("[WARN] No documents in", self.data_path or self.sources)
                return

            texts = []
            records = []

            for doc_idx, doc in enumerate(documents):
                content = (doc.get("content") or "").strip()
                meta = doc.get("metadata", {})
                filename = meta.get("filename", f"doc_{doc_idx}")
                source = meta.get("source", "")

                if not content:
                    continue

                if len(content) > CHUNK_SIZE:
                    # Split into overlapping chunks
                    for chunk_idx, start in enumerate(
                        range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP)
                    ):
                        chunk = content[start : start + CHUNK_SIZE]
                        if not chunk.strip():
                            continue
                        text = f"Document '{filename}' (Part {chunk_idx + 1}): {chunk}"
                        texts.append(text)
                        records.append({
                            "type": "doc_chunk",
                            "id": f"{doc_idx}_c{chunk_idx}",
                            "data": {
                                "filename": filename,
                                "source": source,
                                "chunk_index": chunk_idx,
                                "full_text": chunk,
                            },
                            "text": text,
                        })
                else:
                    text = f"Document '{filename}': {content}"
                    texts.append(text)
                    records.append({
                        "type": "doc_full",
                        "id": f"doc_{doc_idx}",
                        "data": {
                            "filename": filename,
                            "source": source,
                            "full_text": content,
                        },
                        "text": text,
                    })

            if not texts:
                return

            print("Computing embeddings...")
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=True)
            self.embeddings_cache = {
                "texts": texts,
                "embeddings": embeddings,
                "data": records,
            }
            from_label = "sources" if self.sources is not None else self.data_path
            print(f"[OK] {len(embeddings)} embeddings from {from_label}")

        except Exception as e:
            print(f"[FAIL] RAG load: {e}")
            self.embeddings_cache = {}

    def retrieve_relevant_context(self, question: str, top_k: int = 3) -> str:
        """
        Retrieve the most relevant document chunks for a question.
        Returns concatenated context text.
        """
        if not self.embedding_model or not self.embeddings_cache:
            return ""

        try:
            question_embedding = self.embedding_model.encode(
                question, convert_to_tensor=True
            )
            similarities = cosine_similarity(
                question_embedding.reshape(1, -1),
                self.embeddings_cache["embeddings"],
            )[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            context_parts = []
            for idx in top_indices:
                if similarities[idx] <= 0.001:
                    continue
                record = self.embeddings_cache["data"][idx]
                data = record["data"]
                if record["type"] == "doc_chunk":
                    context_parts.append(
                        f"Document '{data['filename']}' (Part {data['chunk_index'] + 1}):\n"
                        f"{data['full_text']}"
                    )
                else:
                    context_parts.append(
                        f"Document '{data['filename']}':\n{data['full_text']}"
                    )

            if not context_parts and len(top_indices) > 0:
                data = self.embeddings_cache["data"][top_indices[0]]["data"]
                snippet = (data.get("full_text") or "")[:500]
                context_parts.append(f"Document '{data['filename']}':\n{snippet}...")

            return "\n\n".join(context_parts)

        except Exception:
            return ""

    def enhance_prompt_with_rag(
        self,
        question: str,
        original_context: str = "",
        top_k: int = 5,
    ) -> Tuple[str, str]:
        """
        Get RAG context for a question and return enhanced context plus status.
        """
        context = self.retrieve_relevant_context(question, top_k=top_k)
        if context:
            enhanced = f"Document context:\n{original_context}\n\n{context}"
            return enhanced, "RAG-enhanced"
        return f"Document context:\n{original_context}", "No RAG context"

    def get_rag_stats(self) -> Dict[str, Any]:
        """Return statistics about the RAG service state."""
        if not self.embeddings_cache:
            return {
                "embedding_model_loaded": self.embedding_model is not None,
                "cached_embeddings": 0,
                "documents": 0,
                "data_path": self.data_path,
            }
        doc_count = sum(
            1
            for r in self.embeddings_cache["data"]
            if r["type"].startswith("doc")
        )
        return {
            "embedding_model_loaded": self.embedding_model is not None,
            "cached_embeddings": len(self.embeddings_cache.get("embeddings", [])),
            "documents": doc_count,
            "data_path": self.data_path,
        }


def create_rag_service(
    data_path: str = "data",
    sources: Optional[List[Any]] = None,
) -> RAGService:
    """Create and return a RAGService instance."""
    return RAGService(data_path=data_path, sources=sources)
