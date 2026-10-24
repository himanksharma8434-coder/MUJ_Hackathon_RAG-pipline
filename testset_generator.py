import json
import re
from dataclasses import dataclass
from typing import List, Optional
from rag_pipeline import Chunk


@dataclass
class SyntheticTestCase:
    id: str
    question: str
    ground_truth: str
    source_chunk_id: str
    question_type: str  # "Factual" | "Numeric/Policy" | "Reasoning"


class SyntheticTestsetGenerator:
    """
    Deliverable 2: Automatically synthesizes evaluation questions + ground-truth answers
    from uploaded document chunks.
    """

    @staticmethod
    def generate_testset(
        chunks: List[Chunk],
        num_questions: int = 5,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ) -> List[SyntheticTestCase]:
        if not chunks:
            return []

        if openai_api_key:
            return SyntheticTestsetGenerator._generate_with_llm(
                chunks, num_questions, openai_api_key, model
            )
        return SyntheticTestsetGenerator._generate_locally(chunks, num_questions)

    @staticmethod
    def _generate_locally(chunks: List[Chunk], num_questions: int) -> List[SyntheticTestCase]:
        """
        Deterministic rule-based extractor that turns high-information sentences
        (containing numbers, policies, definitions, or requirements) into Q&A pairs.
        """
        test_cases: List[SyntheticTestCase] = []
        q_idx = 1

        for chunk in chunks:
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", chunk.text)
                if len(s.strip()) > 35
            ]
            for sent in sentences:
                if len(test_cases) >= num_questions:
                    break

                # Pattern 1: Sentence with numbers / percentages / durations / dollar amounts
                num_match = re.search(
                    r"\b(\d+(?:\.\d+)?(?:\s*%|\s*hours|\s*days|\s*minutes|\s*seconds|\s*ms|\s*GB|\s*USD|\$)?)\b",
                    sent,
                    re.IGNORECASE,
                )
                # Extract leading subject phrase
                words = sent.split()
                subject = " ".join(words[: min(6, len(words))]).rstrip(",.:;")

                if num_match:
                    question = f"What specific metric, limit, or requirement applies to '{subject}'?"
                    q_type = "Numeric/Policy"
                elif any(k in sent.lower() for k in ["must", "required", "policy", "protocol", "responsible"]):
                    question = f"What is the policy or requirement regarding '{subject}'?"
                    q_type = "Reasoning"
                else:
                    question = f"According to the documentation, what is stated about '{subject}'?"
                    q_type = "Factual"

                test_cases.append(
                    SyntheticTestCase(
                        id=f"TEST-{q_idx:02d}",
                        question=question,
                        ground_truth=sent,
                        source_chunk_id=chunk.chunk_id,
                        question_type=q_type,
                    )
                )
                q_idx += 1
                break  # Pick one diverse question per chunk first

        return test_cases

    @staticmethod
    def _generate_with_llm(
        chunks: List[Chunk], num_questions: int, openai_api_key: str, model: str
    ) -> List[SyntheticTestCase]:
        from openai import OpenAI

        client = OpenAI(api_key=openai_api_key)
        selected_chunks = chunks[: min(len(chunks), num_questions)]
        payload = [{"chunk_id": c.chunk_id, "text": c.text} for c in selected_chunks]

        prompt = f"""Generate {num_questions} diverse evaluation test questions from the following document chunks.
Each question must be answerable using its source chunk.
Return ONLY valid JSON with key "test_cases" containing a list of objects:
- "question": str
- "ground_truth": str (concise, complete factual answer directly supported by the chunk)
- "source_chunk_id": str
- "question_type": "Factual" | "Numeric/Policy" | "Reasoning"

Chunks:
{json.dumps(payload)}"""

        resp = client.chat.completions.create(
            model=model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        data = json.loads(resp.choices[0].message.content)
        cases = []
        for idx, item in enumerate(data.get("test_cases", [])[:num_questions], start=1):
            cases.append(
                SyntheticTestCase(
                    id=f"TEST-{idx:02d}",
                    question=item["question"],
                    ground_truth=item["ground_truth"],
                    source_chunk_id=item["source_chunk_id"],
                    question_type=item.get("question_type", "Factual"),
                )
            )
        return cases


if __name__ == "__main__":
    from rag_pipeline import DecoupledRAGPipeline

    print("=" * 60)
    print("Testing Synthetic Test-Set Generation (Deliverable 2)")
    print("=" * 60)

    sample_docs = [
        {
            "name": "hr_policy.pdf",
            "content": (
                "Employees are entitled to 20 days of paid annual leave per calendar year. "
                "Parental leave covers up to 16 weeks of fully compensated leave for primary caregivers."
            ),
        },
        {
            "name": "finance_policy.pdf",
            "content": (
                "Any expense reports exceeding $500 USD must be submitted within 14 days and requires approval from the Zurich Compliance Board. "
                "Mileage reimbursement is calculated at 0.65 USD per mile traveled for official business."
            ),
        },
        {
            "name": "security_protocol.pdf",
            "content": (
                "All remote employees must connect to the corporate VPN when accessing internal databases. "
                "Multi-factor authentication (MFA) is strictly required for all administrative logins."
            ),
        },
        {
            "name": "infrastructure_sla.pdf",
            "content": (
                "Data backups are conducted every 24 hours to ensure high availability and disaster recovery. "
                "System uptime SLA is guaranteed at 99.95% availability excluding scheduled maintenance."
            ),
        },
    ]

    pipeline = DecoupledRAGPipeline(chunk_size=100, chunk_overlap=20)
    num_chunks = pipeline.ingest_documents(sample_docs)
    print(f"Successfully ingested sample documents into {num_chunks} chunks.")

    print("\nGenerating synthetic Q&A test cases (local rule-based mode)...")
    test_cases = SyntheticTestsetGenerator.generate_testset(pipeline.chunks, num_questions=4)

    print(f"Generated {len(test_cases)} test cases:\n")
    for tc in test_cases:
        print(f"[{tc.id}] Type: {tc.question_type}")
        print(f"  Source Chunk : {tc.source_chunk_id}")
        print(f"  Question     : {tc.question}")
        print(f"  Ground Truth : {tc.ground_truth}")
        print("-" * 60)