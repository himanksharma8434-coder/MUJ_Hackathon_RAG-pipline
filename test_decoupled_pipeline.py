from evaluator import RAGEvaluator
from rag_pipeline import DecoupledRAGPipeline
from test_set_generator import SyntheticTestsetGenerator


DOCS = [
    {
        "name": "sla.txt",
        "content": (
            "Production database clusters guarantee 99.95% monthly uptime. "
            "If monthly uptime falls below 99.95%, customers receive a 15% service credit "
            "on their monthly bill. Critical incidents require a response within 15 minutes."
        ),
    },
    {
        "name": "security.txt",
        "content": (
            "Privileged SSH keys must rotate every 90 days. "
            "Unauthorized API key leaks must be revoked within 10 minutes and reported within 4 hours."
        ),
    },
    {
        "name": "finance.txt",
        "content": (
            "Domestic travel meal reimbursement is capped at $75 per day. "
            "Receipts over $25 must be uploaded within 21 days of trip completion."
        ),
    },
    {
        "name": "access.txt",
        "content": (
            "Multi-factor authentication is required for all administrative logins. "
            "Remote employees must connect to the corporate VPN for internal databases."
        ),
    },
]
QUERY = "What is the guaranteed monthly uptime and service credit?"
GROUND_TRUTH = (
    "Production database clusters guarantee 99.95% monthly uptime, and customers receive "
    "a 15% service credit if uptime falls below 99.95%."
)


def make_pipeline() -> DecoupledRAGPipeline:
    pipeline = DecoupledRAGPipeline(chunk_size=65, chunk_overlap=12)
    assert pipeline.ingest_documents(DOCS) >= 4
    return pipeline


def test_normal_run_is_a_healthy_pass():
    pipeline = make_pipeline()
    chunks = pipeline.retrieve(QUERY, top_k=2)
    answer = pipeline.generate(QUERY, chunks)
    result = RAGEvaluator().evaluate(QUERY, chunks, answer, GROUND_TRUTH)

    assert result.root_cause.verdict_code == "PASS"
    assert result.context_relevance >= 0.65
    assert result.context_recall >= 0.65
    assert result.faithfulness >= 0.65
    assert result.answer_accuracy >= 0.65


def test_wrong_chunks_are_diagnosed_as_retriever_failure():
    pipeline = make_pipeline()
    chunks = pipeline.retrieve(QUERY, top_k=2, simulate_bad_retrieval=True)
    answer = pipeline.generate(QUERY, chunks)
    result = RAGEvaluator().evaluate(QUERY, chunks, answer, GROUND_TRUTH)

    assert result.root_cause.verdict_code == "RETRIEVER_FAILURE"
    assert result.faithfulness == 1.0
    assert any(claim.status == "ABSTENTION" for claim in result.claim_verifications)


def test_hallucinated_claims_are_attributed_to_generator():
    pipeline = make_pipeline()
    chunks = pipeline.retrieve(QUERY, top_k=2)
    answer = pipeline.generate(QUERY, chunks, simulate_mode="hallucinate")
    result = RAGEvaluator().evaluate(QUERY, chunks, answer, GROUND_TRUTH)

    assert result.root_cause.verdict_code == "GENERATOR_HALLUCINATION"
    assert result.faithfulness < 0.65
    assert any(claim.status == "HALLUCINATED" for claim in result.claim_verifications)


def test_omitted_values_are_diagnosed_as_generator_omission():
    pipeline = make_pipeline()
    chunks = pipeline.retrieve(QUERY, top_k=2)
    answer = pipeline.generate(QUERY, chunks, simulate_mode="omit")
    result = RAGEvaluator().evaluate(QUERY, chunks, answer, GROUND_TRUTH)

    assert result.root_cause.verdict_code == "GENERATOR_OMISSION"
    assert result.faithfulness >= 0.65
    assert result.answer_accuracy < 0.65
    assert result.missing_ground_truth_facts


def test_compound_fault_injection_hits_both_components():
    pipeline = make_pipeline()
    chunks = pipeline.retrieve(QUERY, top_k=2, simulate_bad_retrieval=True)
    answer = pipeline.generate(QUERY, chunks, simulate_mode="hallucinate")
    result = RAGEvaluator().evaluate(QUERY, chunks, answer, GROUND_TRUTH)

    assert result.root_cause.verdict_code == "COMPOUND_FAILURE"
    assert result.faithfulness < 0.65


def test_synthetic_cases_are_traceable_and_classified():
    pipeline = make_pipeline()
    cases = SyntheticTestsetGenerator.generate_testset(pipeline.chunks, num_questions=6)

    assert cases
    assert len(cases) <= 6
    assert all(case.source_chunk_id in {chunk.chunk_id for chunk in pipeline.chunks} for case in cases)
    assert all(case.ground_truth for case in cases)
    assert {case.question_type for case in cases} & {"Numeric/Policy", "Reasoning"}


def test_empty_reingest_clears_previous_indexes():
    pipeline = make_pipeline()
    assert pipeline.ingest_documents([]) == 0
    assert pipeline.retrieve("anything") == []
