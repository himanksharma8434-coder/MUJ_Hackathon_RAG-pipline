"""Independent retriever/generator scoring and root-cause diagnostics."""

from dataclasses import dataclass, field
import json
import re
from typing import List, Optional, Sequence, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_pipeline import Chunk


@dataclass
class ClaimVerification:
    claim: str
    status: str  # SUPPORTED | HALLUCINATED | ABSTENTION
    supporting_chunk_id: Optional[str]
    confidence: float
    reason: str


@dataclass
class ChunkEvaluation:
    chunk_id: str
    doc_name: str
    text: str
    relevance_score: float
    is_relevant: bool
    contains_ground_truth: bool


@dataclass
class RootCauseReport:
    verdict_code: str
    verdict_label: str
    severity: str  # success | warning | error
    summary: str
    failed_component: str
    remediation_steps: List[str]


@dataclass
class EvaluationResult:
    query: str
    ground_truth: Optional[str]
    generated_answer: str
    context_relevance: float
    context_recall: Optional[float]
    faithfulness: float
    answer_accuracy: Optional[float]
    chunk_evaluations: List[ChunkEvaluation] = field(default_factory=list)
    claim_verifications: List[ClaimVerification] = field(default_factory=list)
    missing_ground_truth_facts: List[str] = field(default_factory=list)
    root_cause: Optional[RootCauseReport] = None


class RAGEvaluator:
    """Score a RAG run locally or with an optional structured LLM judge."""

    DEFAULT_THRESHOLD = 0.65
    CLAIM_SUPPORT_THRESHOLD = 0.52
    MISSING_FACT_THRESHOLD = 0.55
    STOPWORDS: Set[str] = {
        "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "from", "as", "what", "how",
        "when", "where", "who", "which", "why", "can", "does", "do", "did", "about",
    }
    ABSTENTION_PHRASES = (
        "does not contain enough information",
        "does not contain sufficient information",
        "not enough information",
        "not mentioned in the context",
        "no context was retrieved",
        "cannot answer from the provided context",
        "unable to answer from the provided context",
    )
    NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:\$\s*)?\d[\d,]*(?:\.\d+)?")

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = float(threshold)

    @classmethod
    def _content_tokens(cls, text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"\b[a-z0-9]+\b", str(text).lower())
            if token not in cls.STOPWORDS
        }

    @classmethod
    def _numeric_values(cls, text: str) -> Set[float]:
        values: Set[float] = set()
        for match in cls.NUMBER_PATTERN.findall(str(text)):
            try:
                values.add(float(match.replace("$", "").replace(",", "").strip()))
            except ValueError:
                continue
        return values

    @classmethod
    def _overlap_score(cls, target: str, reference: str) -> float:
        """Return token-recall/TF-IDF overlap with a hard numeric consistency penalty."""

        target_tokens = cls._content_tokens(target)
        reference_tokens = cls._content_tokens(reference)
        if not target_tokens or not reference_tokens:
            return 0.0

        token_recall = len(target_tokens & reference_tokens) / len(target_tokens)
        try:
            matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(
                [str(target), str(reference)]
            )
            cosine = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
        except ValueError:
            cosine = 0.0

        target_numbers = cls._numeric_values(target)
        reference_numbers = cls._numeric_values(reference)
        numeric_penalty = 1.0
        if target_numbers and not target_numbers.issubset(reference_numbers):
            numeric_penalty = 0.35

        raw_score = (0.65 * token_recall + 0.35 * cosine) * numeric_penalty
        return round(max(0.0, min(1.0, raw_score * 1.15)), 3)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not str(text).strip():
            return []
        sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
        return [sentence.strip(" -*\t") for sentence in sentences if sentence.strip(" -*\t")]

    @classmethod
    def _is_abstention(cls, claim: str) -> bool:
        lowered = claim.lower().strip()
        return not lowered or any(phrase in lowered for phrase in cls.ABSTENTION_PHRASES)

    @staticmethod
    def _bounded(value: float, default: float = 0.0) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 3)
        except (TypeError, ValueError):
            return default

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> EvaluationResult:
        """Evaluate a run; fall back to the deterministic judge if LLM judging fails."""

        if openai_api_key:
            try:
                return self._evaluate_with_llm(
                    query,
                    retrieved_chunks,
                    generated_answer,
                    ground_truth,
                    openai_api_key,
                    model,
                )
            except Exception:
                # The zero-key path is the reliable baseline for the dashboard.
                # A transient API or schema failure should not take the app down.
                pass
        return self._evaluate_locally(query, retrieved_chunks, generated_answer, ground_truth)

    def _evaluate_locally(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        generated_answer: str,
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        """Run the zero-cost lexical/TF-IDF evaluation path."""

        query_tokens = self._content_tokens(query)
        chunk_evaluations: List[ChunkEvaluation] = []
        for chunk in retrieved_chunks:
            query_overlap = self._overlap_score(query, chunk.text)
            relevance = self._bounded(max(float(chunk.score), query_overlap))
            contains_ground_truth = bool(
                ground_truth and self._overlap_score(ground_truth, chunk.text) >= self.MISSING_FACT_THRESHOLD
            )
            if contains_ground_truth:
                relevance = max(relevance, 0.85)
            chunk_evaluations.append(
                ChunkEvaluation(
                    chunk_id=chunk.chunk_id,
                    doc_name=chunk.doc_name,
                    text=chunk.text,
                    relevance_score=relevance,
                    is_relevant=relevance >= 0.45 or contains_ground_truth,
                    contains_ground_truth=contains_ground_truth,
                )
            )

        if chunk_evaluations:
            scores = [item.relevance_score for item in chunk_evaluations]
            context_relevance = round(0.7 * max(scores) + 0.3 * sum(scores) / len(scores), 3)
        else:
            context_relevance = 0.0

        combined_context = " ".join(chunk.text for chunk in retrieved_chunks)
        context_recall: Optional[float] = None
        if ground_truth:
            gt_sentences = self._split_sentences(ground_truth)
            recalls = [self._overlap_score(sentence, combined_context) for sentence in gt_sentences]
            context_recall = round(sum(recalls) / len(recalls), 3) if recalls else 0.0

        claim_verifications: List[ClaimVerification] = []
        for claim in self._split_sentences(generated_answer):
            if self._is_abstention(claim):
                claim_verifications.append(
                    ClaimVerification(
                        claim=claim,
                        status="ABSTENTION",
                        supporting_chunk_id=None,
                        confidence=1.0,
                        reason="The answer explicitly abstains because the retrieved context is insufficient.",
                    )
                )
                continue

            best_chunk_id: Optional[str] = None
            best_score = 0.0
            for chunk in retrieved_chunks:
                score = self._overlap_score(claim, chunk.text)
                if score > best_score:
                    best_score = score
                    best_chunk_id = chunk.chunk_id
            combined_score = self._overlap_score(claim, combined_context)
            final_score = max(best_score, combined_score)
            if final_score >= self.CLAIM_SUPPORT_THRESHOLD:
                claim_verifications.append(
                    ClaimVerification(
                        claim=claim,
                        status="SUPPORTED",
                        supporting_chunk_id=best_chunk_id,
                        confidence=final_score,
                        reason=(
                            f"Claim is supported by retrieved context"
                            f"{f' in [{best_chunk_id}]' if best_chunk_id else ''}."
                        ),
                    )
                )
            else:
                claim_verifications.append(
                    ClaimVerification(
                        claim=claim,
                        status="HALLUCINATED",
                        supporting_chunk_id=None,
                        confidence=round(1.0 - final_score, 3),
                        reason="No retrieved chunk supports the claim and its concrete values.",
                    )
                )

        if claim_verifications:
            faithful_claims = sum(
                item.status in {"SUPPORTED", "ABSTENTION"} for item in claim_verifications
            )
            faithfulness = round(faithful_claims / len(claim_verifications), 3)
        else:
            faithfulness = 0.0

        missing_facts: List[str] = []
        if ground_truth:
            gt_sentences = self._split_sentences(ground_truth)
            gt_scores = [self._overlap_score(sentence, generated_answer) for sentence in gt_sentences]
            for sentence, score in zip(gt_sentences, gt_scores):
                if score < self.MISSING_FACT_THRESHOLD:
                    missing_facts.append(sentence)
            raw_accuracy = sum(gt_scores) / max(1, len(gt_scores))
            # Hallucinated claims should reduce end-to-end accuracy, while an
            # abstention remains a faithful (but incomplete) answer.
            answer_accuracy: Optional[float] = round(
                max(0.0, min(1.0, raw_accuracy * (0.7 + 0.3 * faithfulness))), 3
            )
        else:
            query_coverage = len(query_tokens & self._content_tokens(generated_answer)) / max(
                1, len(query_tokens)
            )
            if any(item.status == "ABSTENTION" for item in claim_verifications):
                answer_accuracy = 0.15
            else:
                answer_accuracy = round(
                    max(0.0, min(1.0, 0.5 * query_coverage + 0.5 * faithfulness)), 3
                )

        root_cause = self.diagnose_root_cause(
            context_relevance=context_relevance,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_accuracy=answer_accuracy,
        )
        return EvaluationResult(
            query=query,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            context_relevance=context_relevance,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_accuracy=answer_accuracy,
            chunk_evaluations=chunk_evaluations,
            claim_verifications=claim_verifications,
            missing_ground_truth_facts=missing_facts,
            root_cause=root_cause,
        )

    def _evaluate_with_llm(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        generated_answer: str,
        ground_truth: Optional[str],
        openai_api_key: str,
        model: str,
    ) -> EvaluationResult:
        """Use an OpenAI JSON judge while preserving the same result schema."""

        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        payload = [{"chunk_id": chunk.chunk_id, "text": chunk.text} for chunk in retrieved_chunks]
        prompt = f"""You are a strict decoupled RAG evaluator. Evaluate retrieval separately from generation.
Return only valid JSON matching this schema:
{{
  "context_relevance": number,
  "context_recall": number or null,
  "faithfulness": number,
  "answer_accuracy": number or null,
  "chunk_evaluations": [{{"chunk_id": string, "relevance_score": number, "is_relevant": boolean, "contains_ground_truth": boolean}}],
  "claim_verifications": [{{"claim": string, "status": "SUPPORTED"|"HALLUCINATED"|"ABSTENTION", "supporting_chunk_id": string or null, "confidence": number, "reason": string}}],
  "missing_ground_truth_facts": [string]
}}

Query: {query}
Ground truth: {ground_truth or "N/A"}
Retrieved chunks: {json.dumps(payload)}
Generated answer: {generated_answer}

Scores must be between 0 and 1. Faithfulness must compare the answer only with retrieved chunks.
"""
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        chunk_map = {chunk.chunk_id: chunk for chunk in retrieved_chunks}

        chunk_evaluations = []
        for item in data.get("chunk_evaluations", []):
            original = chunk_map.get(item.get("chunk_id"))
            if original is None:
                continue
            chunk_evaluations.append(
                ChunkEvaluation(
                    chunk_id=original.chunk_id,
                    doc_name=original.doc_name,
                    text=original.text,
                    relevance_score=self._bounded(item.get("relevance_score", 0.0)),
                    is_relevant=bool(item.get("is_relevant", False)),
                    contains_ground_truth=bool(item.get("contains_ground_truth", False)),
                )
            )

        allowed_statuses = {"SUPPORTED", "HALLUCINATED", "ABSTENTION"}
        claim_verifications = []
        for item in data.get("claim_verifications", []):
            status = str(item.get("status", "HALLUCINATED")).upper()
            if status not in allowed_statuses:
                status = "HALLUCINATED"
            claim_verifications.append(
                ClaimVerification(
                    claim=str(item.get("claim", "")),
                    status=status,
                    supporting_chunk_id=item.get("supporting_chunk_id"),
                    confidence=self._bounded(item.get("confidence", 0.0)),
                    reason=str(item.get("reason", "")),
                )
            )

        context_relevance = self._bounded(data.get("context_relevance", 0.0))
        context_recall = (
            None
            if data.get("context_recall") is None
            else self._bounded(data.get("context_recall", 0.0))
        )
        faithfulness = self._bounded(data.get("faithfulness", 0.0))
        answer_accuracy = (
            None
            if data.get("answer_accuracy") is None
            else self._bounded(data.get("answer_accuracy", 0.0))
        )
        return EvaluationResult(
            query=query,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            context_relevance=context_relevance,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_accuracy=answer_accuracy,
            chunk_evaluations=chunk_evaluations,
            claim_verifications=claim_verifications,
            missing_ground_truth_facts=[
                str(item) for item in data.get("missing_ground_truth_facts", [])
            ],
            root_cause=self.diagnose_root_cause(
                context_relevance,
                context_recall,
                faithfulness,
                answer_accuracy,
            ),
        )

    def diagnose_root_cause(
        self,
        context_relevance: float,
        context_recall: Optional[float],
        faithfulness: float,
        answer_accuracy: Optional[float],
    ) -> RootCauseReport:
        """Apply the five-row decision matrix from the product specification."""

        retriever_score = min(context_relevance, context_recall) if context_recall is not None else context_relevance
        accuracy_score = answer_accuracy if answer_accuracy is not None else faithfulness
        retriever_ok = retriever_score >= self.threshold
        faithfulness_ok = faithfulness >= self.threshold
        accuracy_ok = accuracy_score >= self.threshold

        if retriever_ok and faithfulness_ok and accuracy_ok:
            return RootCauseReport(
                "PASS",
                "Healthy End-to-End Execution",
                "success",
                f"Relevant context ({retriever_score:.0%}) supported an accurate ({accuracy_score:.0%}) and grounded ({faithfulness:.0%}) answer.",
                "None",
                ["No action needed. Pipeline is operating within target thresholds."],
            )
        if not retriever_ok and faithfulness_ok:
            return RootCauseReport(
                "RETRIEVER_FAILURE",
                "Retriever Failure — Wrong or Missing Context",
                "error",
                f"The Retriever scored {retriever_score:.0%}; the Generator remained faithful ({faithfulness:.0%}) to the available context or abstained.",
                "Retriever",
                [
                    "Increase top-k or add a cross-encoder reranker.",
                    "Tune chunk size and overlap so key facts remain together.",
                    "Add query expansion or adjust BM25/dense retrieval weights.",
                ],
            )
        if retriever_ok and not faithfulness_ok:
            return RootCauseReport(
                "GENERATOR_HALLUCINATION",
                "Generator Failure — Hallucination Despite Good Context",
                "error",
                f"The Retriever succeeded ({retriever_score:.0%}), but unsupported claims reduced Faithfulness to {faithfulness:.0%}.",
                "Generator",
                [
                    "Enforce a strict answer-only-from-context prompt with chunk citations.",
                    "Lower generation temperature to 0.0–0.1.",
                    "Add a post-generation NLI or claim verifier before returning the answer.",
                ],
            )
        if retriever_ok and faithfulness_ok and not accuracy_ok:
            return RootCauseReport(
                "GENERATOR_OMISSION",
                "Generator Failure — Incomplete Extraction",
                "warning",
                f"The Retriever found the needed context ({retriever_score:.0%}) and the answer stayed grounded ({faithfulness:.0%}), but Accuracy was only {accuracy_score:.0%}.",
                "Generator",
                [
                    "Ask the Generator to enumerate every matching numerical and factual detail.",
                    "Place high-relevance chunks at the beginning or end to reduce lost-in-the-middle omissions.",
                ],
            )
        return RootCauseReport(
            "COMPOUND_FAILURE",
            "Compound Failure — Bad Retrieval + Generator Hallucination",
            "error",
            f"The Retriever scored {retriever_score:.0%} and the Generator was not faithful ({faithfulness:.0%}); both stages require remediation.",
            "Retriever + Generator",
            [
                "Retriever: improve indexing, chunking, query expansion, and reranking.",
                "Generator: abstain when context is insufficient and verify every claim against retrieved chunks.",
            ],
        )
