"""Retrieval and generation primitives for the decoupled RAG demo.

The module intentionally keeps retrieval and generation separate.  That makes it
possible for the evaluator to grade the retrieved context before looking at the
answer, which is the key property of the diagnostic pipeline.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    """A document chunk and its query-specific retrieval score."""

    chunk_id: str
    doc_name: str
    text: str
    score: float = 0.0


class DecoupledRAGPipeline:
    """Chunk documents, retrieve context, and generate a grounded answer.

    Retrieval uses a 50/50 hybrid of BM25 and TF-IDF cosine similarity.  The
    local generator is deterministic so the application is useful without an
    API key; an optional OpenAI path is available for deeper demonstrations.
    """

    VALID_SIMULATION_MODES = {"normal", "hallucinate", "omit"}

    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", str(text).lower())

    @staticmethod
    def _normalise_docs(docs: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        normalised: List[Dict[str, str]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            name = str(doc.get("name", "document"))
            content = doc.get("content", "")
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            content = str(content).strip()
            if content:
                normalised.append({"name": name, "content": content})
        return normalised

    def ingest_documents(self, docs: Iterable[Dict[str, str]]) -> int:
        """Ingest ``{'name': ..., 'content': ...}`` documents and build indexes."""

        # Reset every index even when the replacement corpus is empty.  This
        # avoids accidentally retrieving from a previous upload.
        self.chunks = []
        self.bm25 = None
        self.tfidf_matrix = None

        step = self.chunk_size - self.chunk_overlap
        for doc in self._normalise_docs(docs):
            words = doc["content"].split()
            for offset, start in enumerate(range(0, len(words), step), start=1):
                chunk_text = " ".join(words[start : start + self.chunk_size]).strip()
                if len(chunk_text) < 30:
                    continue
                self.chunks.append(
                    Chunk(
                        chunk_id=f"{doc['name']}#C{offset}",
                        doc_name=doc["name"],
                        text=chunk_text,
                    )
                )

        if self.chunks:
            tokenized = [self._tokenize(chunk.text) for chunk in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
            self.tfidf_matrix = self.tfidf.fit_transform(
                [chunk.text for chunk in self.chunks]
            )
        return len(self.chunks)

    @staticmethod
    def _normalise_scores(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return scores
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum <= minimum:
            return np.zeros_like(scores)
        return (scores - minimum) / (maximum - minimum)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        simulate_bad_retrieval: bool = False,
    ) -> List[Chunk]:
        """Retrieve the highest-ranked chunks, or lowest-ranked chunks for a fault demo."""

        if not self.chunks or not str(query).strip() or self.bm25 is None:
            return []
        limit = max(1, min(int(top_k), len(self.chunks)))

        q_tokens = self._tokenize(query)
        bm25_scores = self._normalise_scores(np.asarray(self.bm25.get_scores(q_tokens)))
        q_vec = self.tfidf.transform([query])
        cosine_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        hybrid_scores = 0.5 * bm25_scores + 0.5 * cosine_scores

        if simulate_bad_retrieval:
            ranked_indices = np.argsort(hybrid_scores, kind="stable")[:limit]
        else:
            ranked_indices = np.argsort(-hybrid_scores, kind="stable")[:limit]

        return [
            Chunk(
                chunk_id=self.chunks[int(index)].chunk_id,
                doc_name=self.chunks[int(index)].doc_name,
                text=self.chunks[int(index)].text,
                score=round(float(hybrid_scores[int(index)]), 3),
            )
            for index in ranked_indices
        ]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
            if sentence.strip()
        ]

    @staticmethod
    def _strip_specific_values(text: str) -> str:
        """Remove concrete values while retaining a context-supported statement."""

        without_values = re.sub(
            r"(?:\$\s*)?\b\d[\d,]*(?:\.\d+)?\s*(?:%|hours?|days?|minutes?|seconds?|weeks?|months?|years?|ms|gb|usd)?\b",
            "",
            text,
            flags=re.IGNORECASE,
        )
        without_values = re.sub(r"\s+([,.;:])", r"\1", without_values)
        without_values = re.sub(r"\s{2,}", " ", without_values)
        return without_values.strip(" ,;:")

    def _local_answer(self, query: str, retrieved_chunks: List[Chunk], simulate_mode: str) -> str:
        if not retrieved_chunks:
            return "The provided context does not contain enough information to answer this question."

        query_tokens = set(self._tokenize(query)) - {
            "what", "how", "when", "where", "who", "why", "is", "are", "the",
            "a", "an", "in", "of", "for", "to", "does", "do", "can", "and",
        }
        candidates = []
        for chunk in retrieved_chunks:
            for sentence in self._split_sentences(chunk.text):
                if len(sentence) < 15:
                    continue
                sentence_tokens = set(self._tokenize(sentence))
                overlap = len(query_tokens & sentence_tokens) / max(1, len(query_tokens))
                candidates.append((overlap, sentence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [sentence for score, sentence in candidates[:2] if score >= 0.15]
        if not selected:
            # This is a deliberate abstention.  Do not echo irrelevant context,
            # otherwise the evaluator would correctly classify the echo as a claim.
            if simulate_mode == "hallucinate":
                return (
                    "The provided context does not contain enough information to answer this question. "
                    "Additionally, campus policy U-9942 imposes a 50-point exam penalty. "
                    "It also requires a 72-hour enrollment freeze managed by the Registrar."
                )
            return "The provided context does not contain enough information to answer this question."

        answer = " ".join(selected)
        if simulate_mode == "omit":
            omitted = [self._strip_specific_values(sentence) for sentence in selected]
            omitted = [sentence for sentence in omitted if len(sentence) >= 15]
            return " ".join(omitted) or "The provided context discusses the requested policy."
        if simulate_mode == "hallucinate":
            return (
                f"{answer} Additionally, campus policy U-9942 imposes a 50-point exam penalty. "
                "It also requires a 72-hour enrollment freeze managed by the Registrar."
            )
        return answer

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        simulate_mode: str = "normal",
    ) -> str:
        """Generate an answer from retrieved context with optional fault injection."""

        if simulate_mode not in self.VALID_SIMULATION_MODES:
            raise ValueError(
                f"simulate_mode must be one of {sorted(self.VALID_SIMULATION_MODES)}"
            )

        context_block = "\n\n".join(
            f"[{chunk.chunk_id}]: {chunk.text}" for chunk in retrieved_chunks
        )

        if openai_api_key:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)
            if simulate_mode == "hallucinate":
                system_prompt = (
                    "Answer using the context, then add one confident statistic or policy that is "
                    "not present in the context. This is a controlled hallucination test."
                )
            elif simulate_mode == "omit":
                system_prompt = (
                    "Answer using the context but omit concrete numbers, names, dates, and specific values. "
                    "Remain factually grounded in the remaining wording."
                )
            else:
                system_prompt = (
                    "Answer ONLY from the provided context. If it does not contain enough information, "
                    "say exactly: The provided context does not contain enough information to answer this question."
                )
            response = client.chat.completions.create(
                model=model,
                temperature=0.0 if simulate_mode == "normal" else 0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
                ],
            )
            return (response.choices[0].message.content or "").strip()

        return self._local_answer(query, retrieved_chunks, simulate_mode)
