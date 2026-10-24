"""Deterministic and optional LLM-backed synthetic benchmark generation."""

from dataclasses import dataclass
import json
import re
from typing import List, Optional, Sequence

from rag_pipeline import Chunk


@dataclass(frozen=True)
class SyntheticTestCase:
    id: str
    question: str
    ground_truth: str
    source_chunk_id: str
    question_type: str  # Factual | Numeric/Policy | Reasoning


class SyntheticTestsetGenerator:
    """Create traceable Q&A cases directly from indexed chunks."""

    NUMBER_PATTERN = re.compile(
        r"(?:\$\s*)?\b\d[\d,]*(?:\.\d+)?\s*(?:%|hours?|days?|minutes?|seconds?|weeks?|months?|years?|ms|gb|usd)?\b",
        flags=re.IGNORECASE,
    )
    POLICY_KEYWORDS = {
        "must", "required", "mandatory", "shall", "sla", "policy", "prohibited",
        "approval", "within", "limit", "capped", "penalty", "guarantee", "retained",
    }
    REASONING_KEYWORDS = {
        "must", "required", "mandatory", "shall", "unless", "only", "when", "if",
        "approval", "responsible", "protocol", "condition",
    }

    @classmethod
    def _sentences(cls, text: str) -> List[str]:
        sentences = []
        for raw_sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            sentence = raw_sentence.strip(" -*\t")
            # Word-overlap can expose a trailing fragment from an adjacent
            # chunk.  Only use sentence-like spans as verified ground truth.
            if len(sentence) >= 35 and sentence[:1].isupper():
                sentences.append(sentence)
        return sentences

    @classmethod
    def _question_type(cls, sentence: str) -> str:
        lowered = sentence.lower()
        if cls.NUMBER_PATTERN.search(sentence) or any(
            keyword in lowered for keyword in {"sla", "limit", "capped", "penalty", "uptime", "credit"}
        ):
            return "Numeric/Policy"
        if any(keyword in lowered for keyword in cls.REASONING_KEYWORDS):
            return "Reasoning"
        return "Factual"

    @staticmethod
    def _subject(sentence: str) -> str:
        words = sentence.split()
        subject = " ".join(words[: min(8, len(words))]).rstrip(" ,.:;\n")
        return subject or "the documented topic"

    @classmethod
    def _question_for(cls, sentence: str, question_type: str) -> str:
        subject = cls._subject(sentence)
        if question_type == "Numeric/Policy":
            return f"What metric, limit, duration, or policy requirement applies to '{subject}'?"
        if question_type == "Reasoning":
            return f"What condition or requirement is documented for '{subject}'?"
        return f"According to the documentation, what is stated about '{subject}'?"

    @classmethod
    def _candidate_sentences(cls, chunks: Sequence[Chunk]) -> List[tuple[Chunk, str, int]]:
        candidates: List[tuple[Chunk, str, int]] = []
        for chunk in chunks:
            for sentence in cls._sentences(chunk.text):
                lowered = sentence.lower()
                score = 0
                if cls.NUMBER_PATTERN.search(sentence):
                    score += 4
                score += sum(keyword in lowered for keyword in cls.POLICY_KEYWORDS)
                # Keep ordinary factual sentences eligible, but prioritize
                # sentences that make useful compliance benchmarks.
                score += min(2, len(sentence.split()) // 45)
                candidates.append((chunk, sentence, score))
        return candidates

    @classmethod
    def _generate_locally(cls, chunks: Sequence[Chunk], num_questions: int) -> List[SyntheticTestCase]:
        if num_questions <= 0:
            return []

        candidates = cls._candidate_sentences(chunks)
        selected: List[tuple[Chunk, str, int]] = []
        seen_sentences = set()

        # First take at most one high-information sentence per chunk.  This
        # gives uploads containing several documents a diverse benchmark while
        # preserving upload/chunk order for a predictable first demo question.
        for chunk in chunks:
            chunk_candidates = [item for item in candidates if item[0] == chunk]
            for candidate in chunk_candidates:
                _, sentence, _ = candidate
                key = re.sub(r"\W+", " ", sentence.lower()).strip()
                if key in seen_sentences:
                    continue
                selected.append(candidate)
                seen_sentences.add(key)
                break
            if len(selected) >= num_questions:
                break

        # If the requested size exceeds the number of chunks, fill from the
        # remaining sentences without repeating the same source sentence.
        if len(selected) < num_questions:
            remaining = sorted(candidates, key=lambda item: item[2], reverse=True)
            for candidate in remaining:
                key = re.sub(r"\W+", " ", candidate[1].lower()).strip()
                if key in seen_sentences:
                    continue
                selected.append(candidate)
                seen_sentences.add(key)
                if len(selected) >= num_questions:
                    break

        return [
            SyntheticTestCase(
                id=f"TEST-{index:02d}",
                question=cls._question_for(sentence, cls._question_type(sentence)),
                ground_truth=sentence,
                source_chunk_id=chunk.chunk_id,
                question_type=cls._question_type(sentence),
            )
            for index, (chunk, sentence, _) in enumerate(selected, start=1)
        ]

    @classmethod
    def generate_testset(
        cls,
        chunks: Sequence[Chunk],
        num_questions: int = 5,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> List[SyntheticTestCase]:
        """Generate traceable cases locally, or use an optional LLM judge/generator."""

        if not chunks or num_questions <= 0:
            return []
        if not openai_api_key:
            return cls._generate_locally(chunks, num_questions)

        try:
            return cls._generate_with_llm(chunks, num_questions, openai_api_key, model)
        except Exception:
            return cls._generate_locally(chunks, num_questions)

    @classmethod
    def _generate_with_llm(
        cls,
        chunks: Sequence[Chunk],
        num_questions: int,
        openai_api_key: str,
        model: str,
    ) -> List[SyntheticTestCase]:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        payload = [
            {"chunk_id": chunk.chunk_id, "text": chunk.text}
            for chunk in list(chunks)[: max(num_questions, 8)]
        ]
        prompt = f"""Generate {num_questions} diverse, answerable RAG evaluation cases from these chunks.
Return only JSON with key test_cases. Each item must contain question, ground_truth,
source_chunk_id, and question_type (Factual, Numeric/Policy, or Reasoning).
Ground truth must be directly supported by the cited source chunk.

Chunks:
{json.dumps(payload)}"""
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(response.choices[0].message.content or "{}")
        valid_ids = {chunk.chunk_id for chunk in chunks}
        cases: List[SyntheticTestCase] = []
        for index, item in enumerate(data.get("test_cases", []), start=1):
            if len(cases) >= num_questions:
                break
            source_id = str(item.get("source_chunk_id", ""))
            question = str(item.get("question", "")).strip()
            ground_truth = str(item.get("ground_truth", "")).strip()
            if not question or not ground_truth or source_id not in valid_ids:
                continue
            question_type = str(item.get("question_type", "Factual"))
            if question_type not in {"Factual", "Numeric/Policy", "Reasoning"}:
                question_type = "Factual"
            cases.append(
                SyntheticTestCase(
                    id=f"TEST-{index:02d}",
                    question=question,
                    ground_truth=ground_truth,
                    source_chunk_id=source_id,
                    question_type=question_type,
                )
            )
        return cases or cls._generate_locally(chunks, num_questions)


if __name__ == "__main__":
    from rag_pipeline import DecoupledRAGPipeline

    pipeline = DecoupledRAGPipeline(chunk_size=100, chunk_overlap=20)
    pipeline.ingest_documents(
        [{"name": "demo.txt", "content": "The uptime SLA is 99.95% and incidents must be acknowledged within 15 minutes."}]
    )
    for test_case in SyntheticTestsetGenerator.generate_testset(pipeline.chunks):
        print(test_case)
