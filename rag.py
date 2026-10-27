"""Document loading, chunking, embedding, and similarity retrieval."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Sequence

import numpy as np


class RAGError(Exception):
    """Base error for expected RAG setup or service failures."""


class KnowledgeBaseError(RAGError):
    """Raised when the knowledge folder cannot provide usable documents."""


@dataclass(frozen=True)
class DocumentChunk:
    """A searchable section of one knowledge document."""

    text: str
    source: str
    is_placeholder: bool = False


@dataclass(frozen=True)
class RetrievedChunk:
    """A document chunk and its cosine similarity score."""

    text: str
    source: str
    score: float


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving the document's meaning."""
    return re.sub(r"\s+", " ", text).strip()


def load_documents(knowledge_dir: Path) -> List[tuple[str, str]]:
    """Load every non-empty TXT file, returning (filename, cleaned text)."""
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        raise KnowledgeBaseError(
            f"Knowledge folder was not found: {knowledge_dir}"
        )

    documents: List[tuple[str, str]] = []
    empty_files: List[str] = []
    for path in sorted(knowledge_dir.glob("*.txt")):
        try:
            text = clean_text(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise KnowledgeBaseError(
                f"Could not read knowledge file '{path.name}'."
            ) from exc
        if text:
            documents.append((path.name, text))
        else:
            empty_files.append(path.name)

    if empty_files:
        names = ", ".join(empty_files)
        raise KnowledgeBaseError(
            f"Knowledge file(s) are empty: {names}. Add approved information "
            "before starting the chatbot."
        )

    if not documents:
        raise KnowledgeBaseError(
            "No non-empty .txt files were found in the knowledge folder."
        )
    return documents


def split_into_chunks(
    documents: Sequence[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[DocumentChunk]:
    """Split documents into overlapping word-based chunks."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    chunks: List[DocumentChunk] = []
    step = chunk_size - chunk_overlap
    for source, text in documents:
        words = text.split()
        is_placeholder = text.lower().startswith("placeholder:")
        for start in range(0, len(words), step):
            section = " ".join(words[start : start + chunk_size]).strip()
            if section:
                chunks.append(
                    DocumentChunk(
                        text=section,
                        source=source,
                        is_placeholder=is_placeholder,
                    )
                )
            if start + chunk_size >= len(words):
                break
    if not chunks:
        raise KnowledgeBaseError("The knowledge files did not contain usable text.")
    return chunks


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors so their dot product is cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class RAGRetriever:
    """Creates a searchable in-memory index for the knowledge base."""

    def __init__(
        self,
        embedding_model: object,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        self.embedding_model = embedding_model
        self.chunks = list(chunks)
        self.embeddings = _normalize_rows(embeddings)

    @classmethod
    def build(
        cls,
        knowledge_dir: Path,
        embedding_model_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> "RAGRetriever":
        documents = load_documents(knowledge_dir)
        chunks = split_into_chunks(documents, chunk_size, chunk_overlap)
        try:
            # Import lazily so the Streamlit interface can show a useful setup
            # error if the Windows ML runtime is not installed yet.
            from sentence_transformers import SentenceTransformer

            embedding_model = SentenceTransformer(embedding_model_name)
            vectors = embedding_model.encode(
                [chunk.text for chunk in chunks],
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise RAGError(
                "The local embedding model could not be loaded. "
                "Check your installation and internet connection for the first download."
            ) from exc

        vectors = np.asarray(vectors, dtype=float)
        if len(vectors) != len(chunks) or vectors.size == 0:
            raise RAGError("The local embedding model returned an unexpected result.")
        return cls(embedding_model, chunks, vectors)

    def retrieve(
        self,
        question: str,
        top_k: int = 4,
        threshold: float = 0.35,
    ) -> List[RetrievedChunk]:
        """Return only chunks that meet the configured relevance threshold."""
        question = clean_text(question)
        if not question:
            return []
        try:
            query_vector = self.embedding_model.encode(
                [question],
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except Exception as exc:
            raise RAGError("The local embedding model could not encode the question.") from exc
        query_vector = _normalize_rows(np.asarray(query_vector, dtype=float))[0]
        scores = self.embeddings @ query_vector
        ranked_indexes = np.argsort(scores)[::-1]
        relevant_indexes = [
            index
            for index in ranked_indexes
            if not self.chunks[index].is_placeholder
            and scores[index] >= threshold
        ]

        # Emergency questions should consult the dedicated procedure file first.
        emergency_terms = {
            "emergency",
            "urgent",
            "danger",
            "injury",
            "accident",
            "fire",
            "unsafe",
        }
        if emergency_terms.intersection(set(question.lower().split())):
            emergency_indexes = [
                index
                for index in ranked_indexes
                if self.chunks[index].source == "emergency_procedures.txt"
                and not self.chunks[index].is_placeholder
            ]
            if emergency_indexes and emergency_indexes[0] not in relevant_indexes:
                relevant_indexes.insert(0, emergency_indexes[0])

        return [
            RetrievedChunk(
                text=self.chunks[index].text,
                source=self.chunks[index].source,
                score=float(scores[index]),
            )
            for index in relevant_indexes[:top_k]
        ]
