import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rag_pipeline import Chunk


@dataclass
class ClaimVerification:
    claim: str
    status: str  # "SUPPORTED" (Green) | "HALLUCINATED" (Red) | "ABSTENTION" (Neutral)
    supporting_chunk_id: Optional[str]
    confidence: float
    reason: str


@dataclass
class ChunkEvaluation:
    chunk_id: str
    doc_name: str
    text: str
    relevance_score: float
    is_relevant: bool  # Green if True, Red if False
    contains_ground_truth: bool


@dataclass
class RootCauseReport:
    verdict_code: str  # PASS | RETRIEVER_FAILURE | GENERATOR_HALLUCINATION | GENERATOR_OMISSION | COMPOUND_FAILURE
    verdict_label: str
    severity: str      # "success" | "warning" | "error"
    summary: str
    failed_component: str  # "None" | "Retriever" | "Generator" | "Retriever + Generator"
    remediation_steps: List[str]


@dataclass
class EvaluationResult:
    query: str
    ground_truth: Optional[str]
    generated_answer: str
    # Deliverable 1 Core Metrics (0.0 to 1.0)
    context_relevance: float
    context_recall: Optional[float]
    faithfulness: float
    answer_accuracy: Optional[float]
    # Granular inspection artifacts for Deliverable 4 UI
    chunk_evaluations: List[ChunkEvaluation] = field(default_factory=list)
    claim_verifications: List[ClaimVerification] = field(default_factory=list)
    missing_ground_truth_facts: List[str] = field(default_factory=list)
    # Deliverable 3 Automated Root-Cause Report
    root_cause: Optional[RootCauseReport] = None


class RAGEvaluator:
    """
    Deliverable 1 & 3: Decoupled Retriever and Generator Scoring Backend
    + Automated Root-Cause Analysis Engine.
    Supports both OpenAI LLM-as-a-Judge and a zero-key local lexical/semantic judge.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those", "it", "its", "from", "as", "what", "how",
        "when", "where", "who", "which", "why", "can", "does", "do", "did"
    }

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold

    @classmethod
    def _content_tokens(cls, text: str) -> set:
        tokens = set(re.findall(r"\b[a-z0-9]+\b", text.lower()))
        return tokens - cls.STOPWORDS

    @classmethod
    def _overlap_score(cls, target: str, reference: str) -> float:
        """
        Computes combined token-recall + TF-IDF cosine similarity between `target` and `reference`.
        """
        t_tokens = cls._content_tokens(target)
        r_tokens = cls._content_tokens(reference)
        if not t_tokens or not r_tokens:
            return 0.0

        token_recall = len(t_tokens & r_tokens) / len(t_tokens)

        try:
            vec = TfidfVectorizer().fit_transform([target, reference])
            cos = float(cosine_similarity(vec[0:1], vec[1:2])[0][0])
        except ValueError:
            cos = 0.0

        # Check hard numeric consistency: numbers in target must appear in reference
        target_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", target))
        ref_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", reference))
        num_penalty = 1.0
        if target_nums and not target_nums.issubset(ref_nums):
            num_penalty = 0.35

        raw = (0.65 * token_recall + 0.35 * cos) * num_penalty
        return round(min(1.0, raw * 1.15), 3)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return sents if sents else [text.strip()]

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        generated_answer: str,
        ground_truth: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> EvaluationResult:
        if openai_api_key:
            return self._evaluate_with_llm(
                query, retrieved_chunks, generated_answer, ground_truth, openai_api_key, model
            )
        return self._evaluate_locally(query, retrieved_chunks, generated_answer, ground_truth)

    def _evaluate_locally(
        self,
        query: str,
        retrieved_chunks: List[Chunk],
        generated_answer: str,
        ground_truth: Optional[str] = None,
    ) -> EvaluationResult:
        # 1. RETRIEVER EVALUATION: Context Relevance & Context Recall
        chunk_evals: List[ChunkEvaluation] = []
        q_tokens = self._content_tokens(query)

        for c in retrieved_chunks:
            c_tokens = self._content_tokens(c.text)
            q_hit = len(q_tokens & c_tokens) / max(1, len(q_tokens))
            rel_score = round(min(1.0, max(c.score, q_hit)), 3)
            has_gt = (
                self._overlap_score(ground_truth, c.text) >= 0.55
                if ground_truth
                else False
            )
            if has_gt:
                rel_score = max(rel_score, 0.85)

            chunk_evals.append(
                ChunkEvaluation(
                    chunk_id=c.chunk_id,
                    doc_name=c.doc_name,
                    text=c.text,
                    relevance_score=rel_score,
                    is_relevant=(rel_score >= 0.45 or has_gt),
                    contains_ground_truth=has_gt,
                )
            )

        if chunk_evals:
            # Top-heavy relevance score so high-ranking relevant chunks score well
            top_score = max(ce.relevance_score for ce in chunk_evals)
            mean_score = sum(ce.relevance_score for ce in chunk_evals) / len(chunk_evals)
            context_relevance = round(0.7 * top_score + 0.3 * mean_score, 3)
        else:
            context_relevance = 0.0

        combined_context = " ".join(c.text for c in retrieved_chunks)
        context_recall = None
        if ground_truth:
            gt_sents = self._split_sentences(ground_truth)
            recalls = [self._overlap_score(s, combined_context) for s in gt_sents]
            context_recall = round(sum(recalls) / len(recalls), 3) if recalls else 0.0

        # 2. GENERATOR EVALUATION: Claim-by-Claim Faithfulness (Answer vs Retrieved Chunks ONLY)
        answer_claims = self._split_sentences(generated_answer)
        claim_verifications: List[ClaimVerification] = []

        abstention_phrases = [
            "does not contain enough information",
            "does not contain sufficient information",
            "not mentioned in the context",
            "no context was retrieved",
        ]

        for claim in answer_claims:
            lower_claim = claim.lower()
            if any(p in lower_claim for p in abstention_phrases):
                claim_verifications.append(
                    ClaimVerification(
                        claim=claim,
                        status="ABSTENTION",
                        supporting_chunk_id=None,
                        confidence=1.0,
                        reason="Model faithfully abstained due to missing context.",
                    )
                )
                continue

            best_chunk_id = None
            best_score = 0.0
            for c in retrieved_chunks:
                sc = self._overlap_score(claim, c.text)
                if sc > best_score:
                    best_score = sc
                    best_chunk_id = c.chunk_id

            # Also verify against combined context in case a claim synthesizes two chunks
            combined_score = self._overlap_score(claim, combined_context)
            final_claim_score = max(best_score, combined_score)

            if final_claim_score >= 0.52:
                claim_verifications.append(
                    ClaimVerification(
                        claim=claim,
                        status="SUPPORTED",
                        supporting_chunk_id=best_chunk_id,
                        confidence=final_claim_score,
                        reason=f"Grounded in retrieved chunk [{best_chunk_id}] (score={final_claim_score:.2f}).",
                    )
                )
            else:
                claim_verifications.append(
                    ClaimVerification(
                        claim=claim,
                        status="HALLUCINATED",
                        supporting_chunk_id=None,
                        confidence=round(1.0 - final_claim_score, 3),
                        reason="Claim contains facts/numbers not supported by any retrieved chunk.",
                    )
                )

        if claim_verifications:
            supported_count = sum(
                1 for cv in claim_verifications if cv.status in ("SUPPORTED", "ABSTENTION")
            )
            faithfulness = round(supported_count / len(claim_verifications), 3)
        else:
            faithfulness = 0.0

        # 3. END-TO-END ANSWER ACCURACY (Answer vs Ground Truth, or Query Coverage if no Ground Truth)
        missing_facts: List[str] = []
        if ground_truth:
            gt_sents = self._split_sentences(ground_truth)
            gt_scores = []
            for g_sent in gt_sents:
                sc = self._overlap_score(g_sent, generated_answer)
                gt_scores.append(sc)
                if sc < 0.50:
                    missing_facts.append(g_sent)
            raw_acc = sum(gt_scores) / max(1, len(gt_scores))
            # Penalize accuracy if answer includes hallucinated claims
            answer_accuracy = round(raw_acc * (0.7 + 0.3 * faithfulness), 3)
        else:
            # Reference-free Answer Relevance + Groundedness
            q_cov = len(q_tokens & self._content_tokens(generated_answer)) / max(1, len(q_tokens))
            is_abstaining = any(cv.status == "ABSTENTION" for cv in claim_verifications)
            if is_abstaining:
                answer_accuracy = 0.15
            else:
                answer_accuracy = round(min(1.0, (0.5 * q_cov + 0.5 * faithfulness)), 3)

        # 4. DELIVERABLE 3: Automated Root-Cause Report
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
            chunk_evaluations=chunk_evals,
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
        """
        Uses OpenAI Structured JSON LLM-as-a-Judge to score Retriever and Generator independently.
        """
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        chunks_payload = [
            {"chunk_id": c.chunk_id, "text": c.text} for c in retrieved_chunks
        ]

        prompt = f"""You are a strict RAG Evaluation Judge separating Retriever performance from Generator performance.
Evaluate the following RAG execution and return ONLY valid JSON matching the schema below.

Query: {query}
Ground Truth Answer (optional): {ground_truth or "N/A"}
Retrieved Chunks: {json.dumps(chunks_payload)}
Generated Answer: {generated_answer}

JSON Schema required:
{{
  "context_relevance": float (0.0 to 1.0, how relevant the retrieved chunks are to the query),
  "context_recall": float or null (0.0 to 1.0, proportion of Ground Truth facts present in Retrieved Chunks; null if Ground Truth is N/A),
  "faithfulness": float (0.0 to 1.0, proportion of claims in Generated Answer strictly supported by Retrieved Chunks),
  "answer_accuracy": float (0.0 to 1.0, factual accuracy and completeness of Generated Answer compared to Ground Truth or Query),
  "chunk_evaluations": [
    {{
      "chunk_id": str,
      "relevance_score": float (0.0 to 1.0),
      "is_relevant": bool,
      "contains_ground_truth": bool
    }}
  ],
  "claim_verifications": [
    {{
      "claim": str (exact sentence from Generated Answer),
      "status": "SUPPORTED" | "HALLUCINATED" | "ABSTENTION",
      "supporting_chunk_id": str or null,
      "confidence": float,
      "reason": str
    }}
  ],
  "missing_ground_truth_facts": [str]
}}"""

        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content)

        chunk_map = {c.chunk_id: c for c in retrieved_chunks}
        chunk_evals = []
        for ce in data.get("chunk_evaluations", []):
            orig = chunk_map.get(ce["chunk_id"])
            if orig:
                chunk_evals.append(
                    ChunkEvaluation(
                        chunk_id=orig.chunk_id,
                        doc_name=orig.doc_name,
                        text=orig.text,
                        relevance_score=float(ce.get("relevance_score", 0.0)),
                        is_relevant=bool(ce.get("is_relevant", False)),
                        contains_ground_truth=bool(ce.get("contains_ground_truth", False)),
                    )
                )

        claim_verifications = [
            ClaimVerification(
                claim=cv["claim"],
                status=cv["status"],
                supporting_chunk_id=cv.get("supporting_chunk_id"),
                confidence=float(cv.get("confidence", 1.0)),
                reason=cv.get("reason", ""),
            )
            for cv in data.get("claim_verifications", [])
        ]

        c_rel = float(data.get("context_relevance", 0.0))
        c_rec = (
            float(data["context_recall"])
            if data.get("context_recall") is not None
            else None
        )
        faith = float(data.get("faithfulness", 0.0))
        acc = float(data.get("answer_accuracy", 0.0))

        root_cause = self.diagnose_root_cause(c_rel, c_rec, faith, acc)

        return EvaluationResult(
            query=query,
            ground_truth=ground_truth,
            generated_answer=generated_answer,
            context_relevance=c_rel,
            context_recall=c_rec,
            faithfulness=faith,
            answer_accuracy=acc,
            chunk_evaluations=chunk_evals,
            claim_verifications=claim_verifications,
            missing_ground_truth_facts=data.get("missing_ground_truth_facts", []),
            root_cause=root_cause,
        )

    def diagnose_root_cause(
        self,
        context_relevance: float,
        context_recall: Optional[float],
        faithfulness: float,
        answer_accuracy: Optional[float],
    ) -> RootCauseReport:
        """
        Deliverable 3: Automated Root-Cause Analysis Engine.
        Separates Retriever failures from Generator hallucinations or omissions.
        """
        ret_score = (
            min(context_relevance, context_recall)
            if context_recall is not None
            else context_relevance
        )
        acc_score = answer_accuracy if answer_accuracy is not None else faithfulness

        ret_ok = ret_score >= self.threshold
        faith_ok = faithfulness >= self.threshold
        acc_ok = acc_score >= self.threshold

        if ret_ok and faith_ok and acc_ok:
            return RootCauseReport(
                verdict_code="PASS",
                verdict_label="Healthy End-to-End Execution",
                severity="success",
                summary=(
                    f"Retriever successfully fetched relevant context ({ret_score:.0%}), and Generator "
                    f"produced an accurate ({acc_score:.0%}), fully grounded ({faithfulness:.0%}) response."
                ),
                failed_component="None",
                remediation_steps=["No action needed. Pipeline is operating within target thresholds."],
            )

        if not ret_ok and faith_ok:
            return RootCauseReport(
                verdict_code="RETRIEVER_FAILURE",
                verdict_label="Retriever Failure — Wrong or Missing Context",
                severity="error",
                summary=(
                    f"The Retriever failed to surface the needed chunks (Retriever score: {ret_score:.0%}). "
                    f"The Generator remained faithful ({faithfulness:.0%}) to the poor context (or abstained), "
                    f"causing low final Answer Accuracy ({acc_score:.0%})."
                ),
                failed_component="Retriever",
                remediation_steps=[
                    "Increase `top_k` or add a Cross-Encoder Reranker (e.g., BGE-Reranker / Cohere Rerank).",
                    "Tune chunk size and chunk overlap so key facts are not split across boundaries.",
                    "Add Query Expansion / HyDE or adjust hybrid lexical (BM25) + dense vector weighting.",
                ],
            )

        if ret_ok and not faith_ok:
            return RootCauseReport(
                verdict_code="GENERATOR_HALLUCINATION",
                verdict_label="Generator Failure — Hallucination Despite Good Context",
                severity="error",
                summary=(
                    f"The Retriever succeeded ({ret_score:.0%}), supplying the right chunks. However, "
                    f"the Generator fabricated unsupported claims (Faithfulness: {faithfulness:.0%})."
                ),
                failed_component="Generator",
                remediation_steps=[
                    "Enforce strict grounding in the system prompt ('Answer ONLY using provided chunks; cite [chunk_id]').",
                    "Lower Generator temperature to 0.0 - 0.1 to reduce creative extrapolation.",
                    "Add a post-generation NLI / self-check verifier to strip unsupported sentences.",
                ],
            )

        if ret_ok and faith_ok and not acc_ok:
            return RootCauseReport(
                verdict_code="GENERATOR_OMISSION",
                verdict_label="Generator Failure — Incomplete Extraction (Missed Context Facts)",
                severity="warning",
                summary=(
                    f"The Retriever found the right chunks ({ret_score:.0%}) and the Generator did not hallucinate "
                    f"({faithfulness:.0%}), but the Generator omitted key facts needed to answer the query "
                    f"(Answer Accuracy: {acc_score:.0%})."
                ),
                failed_component="Generator",
                remediation_steps=[
                    "Reorder chunks so highest-relevance passages appear at the very beginning or end (mitigate 'Lost in the Middle').",
                    "Prompt the Generator to extract all matching numerical/factual details before summarizing.",
                ],
            )

        return RootCauseReport(
            verdict_code="COMPOUND_FAILURE",
            verdict_label="Compound Failure — Bad Retrieval + Generator Hallucination",
            severity="error",
            summary=(
                f"Both components failed: the Retriever fetched irrelevant chunks ({ret_score:.0%}), "
                f"and instead of abstaining, the Generator hallucinated from parametric memory (Faithfulness: {faithfulness:.0%})."
            ),
            failed_component="Retriever + Generator",
            remediation_steps=[
                "Retriever Fix: Improve indexing, chunking, and hybrid search recall.",
                "Generator Fix: Add explicit abstention instructions when retrieved chunk relevance is low.",
            ],
        )