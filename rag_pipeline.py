import re
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    text: str
    score: float = 0.0


class DecoupledRAGPipeline:
    """
    Separates Document Chunking & Hybrid Retrieval (BM25 + TF-IDF Cosine)
    from Answer Generation.
    """

    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Okapi] = None
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def ingest_documents(self, docs: List[dict]) -> int:
        """
        Ingests a list of {'name': str, 'content': str} documents and splits into chunks.
        """
        self.chunks = []
        for doc in docs:
            words = doc["content"].split()
            step = max(1, self.chunk_size - self.chunk_overlap)
            idx = 0
            for i in range(0, len(words), step):
                slice_words = words[i : i + self.chunk_size]
                if not slice_words:
                    continue
                chunk_text = " ".join(slice_words).strip()
                if len(chunk_text) > 30:
                    idx += 1
                    self.chunks.append(
                        Chunk(
                            chunk_id=f"{doc['name']}#C{idx}",
                            doc_name=doc["name"],
                            text=chunk_text,
                        )
                    )

        if self.chunks:
            tokenized = [self._tokenize(c.text) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized)
            self.tfidf_matrix = self.tfidf.fit_transform([c.text for c in self.chunks])
        return len(self.chunks)

    def retrieve(
        self, query: str, top_k: int = 3, simulate_bad_retrieval: bool = False
    ) -> List[Chunk]:
        """
        Stage 1: Retriever (Evaluated independently of generation).
        If `simulate_bad_retrieval=True`, deliberately returns the lowest-ranked chunks
        to demonstrate a Retriever Failure in the dashboard.
        """
        if not self.chunks:
            return []

        q_tokens = self._tokenize(query)
        bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=float)
        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()

        q_vec = self.tfidf.transform([query])
        cos_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Hybrid score: 50% lexical BM25 + 50% TF-IDF cosine similarity
        hybrid_scores = 0.5 * bm25_scores + 0.5 * cos_scores

        if simulate_bad_retrieval:
            # Deliberately pick the least relevant chunks to test Retriever Failure detection
            ranked_indices = np.argsort(hybrid_scores)[:top_k]
        else:
            ranked_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            c = self.chunks[int(idx)]
            results.append(
                Chunk(
                    chunk_id=c.chunk_id,
                    doc_name=c.doc_name,
                    text=c.text,
                    score=round(float(hybrid_scores[int(idx)]), 3),
                )
            )
        return results

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        simulate_mode: str = "normal",
    ) -> str:
        """
        Stage 2: Generator (Evaluated independently against retrieved chunks & ground truth).
        `simulate_mode` options:
          - 'normal': Faithful synthesis from retrieved context
          - 'hallucinate': Injects plausible but unsupported claims not in the context
          - 'omit': Ignores key numbers/facts in the retrieved context
        """
        context_block = "\n\n".join(
            [f"[{c.chunk_id}]: {c.text}" for c in retrieved_chunks]
        )

        if openai_api_key:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key)
            if simulate_mode == "hallucinate":
                sys_prompt = (
                    "Answer the user query using the provided context, AND also invent 2 specific, "
                    "confident-sounding statistics, dates, or policies that are NOT mentioned in the context."
                )
            elif simulate_mode == "omit":
                sys_prompt = (
                    "Give an overly vague, 1-sentence high-level response that avoids mentioning any specific "
                    "numbers, names, or concrete answers from the context."
                )
            else:
                sys_prompt = (
                    "You are a precise assistant. Answer the question strictly using ONLY the provided context. "
                    "If the context does not contain the answer, state: 'The provided context does not contain "
                    "enough information to answer this question.'"
                )

            resp = client.chat.completions.create(
                model=model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_block}\n\nQuestion: {query}",
                    },
                ],
            )
            return resp.choices[0].message.content.strip()

        # Local deterministic generator fallback (runs without an API key)
        if not retrieved_chunks:
            return "No context was retrieved to answer this query."

        if simulate_mode == "omit":
            return "The uploaded documentation discusses general system policies and operational guidelines."

        # Extract the most query-relevant sentences from the retrieved chunks
        q_words = set(self._tokenize(query)) - {
            "what", "how", "when", "where", "who", "why", "is", "are", "the", "a", "an", "in", "of", "for", "to", "does"
        }
        candidate_sentences = []
        for c in retrieved_chunks:
            for sent in re.split(r"(?<=[.!?])\s+", c.text):
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                s_words = set(self._tokenize(sent))
                overlap = len(q_words & s_words) / max(1, len(q_words))
                candidate_sentences.append((overlap, sent))

        candidate_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentences = [s for score, s in candidate_sentences[:2] if score >= 0.15]

        if not best_sentences:
            base_answer = (
                "The retrieved context does not contain sufficient information to answer this question. "
                f"It only discusses: {retrieved_chunks[0].text[:140]}..."
            )
        else:
            base_answer = " ".join(best_sentences)

        if simulate_mode == "hallucinate":
            hallucinated_claim = (
                " Additionally, internal audit protocol ISO-9942 mandates a $45,000 penalty "
                "and a 72-hour mandatory freeze managed by the Zurich Compliance Board."
            )
            return base_answer + hallucinated_claim

        return base_answer