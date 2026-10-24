import html
import io
import pandas as pd
import streamlit as st
from pypdf import PdfReader

from rag_pipeline import DecoupledRAGPipeline
from evaluator import RAGEvaluator, EvaluationResult
try:
    from test_set_generator import SyntheticTestsetGenerator
except ImportError:
    from testset_generator import SyntheticTestsetGenerator

st.set_page_config(
    page_title="Decoupled RAG Evaluation Pipeline",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS for clean single-screen layout & Green/Red Scorecards
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
    .scorecard {
        border-radius: 10px;
        padding: 14px 16px;
        color: white;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.10);
        margin-bottom: 8px;
    }
    .scorecard-green { background: linear-gradient(135deg, #16a34a, #15803d); }
    .scorecard-red   { background: linear-gradient(135deg, #dc2626, #b91c1c); }
    .scorecard-gray  { background: linear-gradient(135deg, #475569, #334155); }
    .scorecard-title { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.9; }
    .scorecard-value { font-size: 1.9rem; font-weight: 700; margin: 4px 0; }
    .scorecard-sub   { font-size: 0.76rem; opacity: 0.9; }

    .chunk-box {
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.90rem;
        line-height: 1.45;
    }
    .chunk-green { background-color: #f0fdf4; border-left: 5px solid #16a34a; color: #14532d; }
    .chunk-red   { background-color: #fef2f2; border-left: 5px solid #dc2626; color: #7f1d1d; }

    .claim-green {
        background-color: #dcfce7;
        color: #14532d;
        padding: 2px 6px;
        border-radius: 4px;
        border-bottom: 2px solid #16a34a;
    }
    .claim-red {
        background-color: #fee2e2;
        color: #7f1d1d;
        padding: 2px 6px;
        border-radius: 4px;
        border-bottom: 2px solid #dc2626;
        font-weight: 600;
    }
    .claim-neutral {
        background-color: #fef9c3;
        color: #713f12;
        padding: 2px 6px;
        border-radius: 4px;
        border-bottom: 2px solid #ca8a04;
    }
    .verdict-banner {
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.95rem;
    }
    .verdict-success { background-color: #dcfce7; border: 1px solid #16a34a; color: #14532d; }
    .verdict-error   { background-color: #fee2e2; border: 1px solid #dc2626; color: #7f1d1d; }
    .verdict-warning { background-color: #fef3c7; border: 1px solid #d97706; color: #78350f; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Default sample corpus so the app works immediately on launch
DEFAULT_DOCS = [
    {
        "name": "campus_library_policy.txt",
        "content": (
            "Students may borrow up to 5 books at one time. The standard borrowing period is 14 days. "
            "Students may renew a book once for an additional 7 days if nobody else has requested it. "
            "Books returned late incur a fine of $1 per day per book, capped at $15. "
            "The library is open from 8:00 AM to 10:00 PM on weekdays and from 10:00 AM to 6:00 PM on weekends."
        ),
    },
    {
        "name": "hostel_rules.txt",
        "content": (
            "Hostel quiet hours are from 10:00 PM to 6:00 AM every day. Visitors must sign in at hostel reception "
            "and leave the building by 8:00 PM. Residents must report maintenance problems through the hostel portal "
            "within 24 hours. Cooking with open flames is prohibited inside student rooms."
        ),
    },
    {
        "name": "exam_guidelines.txt",
        "content": (
            "Students should arrive at the examination room at least 30 minutes before the scheduled start time. "
            "A valid university ID is required for every examination. Students arriving more than 15 minutes late "
            "may not be permitted to enter the room. Students must maintain at least 75% attendance to be eligible "
            "for final examinations."
        ),
    },
    {
        "name": "campus_shuttle_schedule.txt",
        "content": (
            "Campus shuttles operate from 7:00 AM to 10:00 PM on weekdays. During normal service, a shuttle arrives "
            "approximately every 20 minutes. Students must show their university ID when boarding after 8:00 PM. "
            "The north route connects the hostel, library, student center, and main gate."
        ),
    },
]


def render_scorecard(title: str, score: float | None, subtitle: str, threshold: float = 0.65):
    if score is None:
        css = "scorecard-gray"
        val_str = "N/A"
        status_str = "No Ground Truth"
    else:
        is_good = score >= threshold
        css = "scorecard-green" if is_good else "scorecard-red"
        val_str = f"{score:.0%}"
        status_str = "PASS (Accurate)" if is_good else "FAIL (Below Threshold)"

    st.markdown(
        f"""
        <div class="scorecard {css}">
            <div class="scorecard-title">{title}</div>
            <div class="scorecard-value">{val_str}</div>
            <div class="scorecard-sub">{status_str} · {subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_uploaded_files(uploaded_files) -> list[dict]:
    docs = []
    for f in uploaded_files:
        try:
            if f.name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(f.read()))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = f.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            st.warning(f"Could not read {f.name}: {exc}")
            continue
        if text.strip():
            docs.append({"name": f.name, "content": text})
    return docs


# Initialize session state
if "pipeline" not in st.session_state:
    pipe = DecoupledRAGPipeline(chunk_size=65, chunk_overlap=12)
    pipe.ingest_documents(DEFAULT_DOCS)
    st.session_state.pipeline = pipe
    st.session_state.synthetic_tests = SyntheticTestsetGenerator.generate_testset(
        pipe.chunks, num_questions=5
    )

# ─── SIDEBAR: Configuration, Document Upload & Fault Injection ───
with st.sidebar:
    st.header("⚙️ Pipeline Controls")
    uploaded_files = st.file_uploader(
        "Upload Documents (.pdf, .txt, .md)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    col_c1, col_c2, col_c3 = st.columns(3)
    chunk_size = col_c1.slider("Chunk Words", 30, 300, 65, step=5)
    top_k = col_c2.slider("Retriever Top-K", 1, 6, 2)
    overlap_max = max(0, min(100, int(chunk_size) - 1))
    chunk_overlap = col_c3.slider(
        "Chunk Overlap", 0, overlap_max, min(12, overlap_max), step=1
    )

    if st.button("📥 Index Documents & Generate Tests", use_container_width=True):
        docs_to_use = parse_uploaded_files(uploaded_files) if uploaded_files else DEFAULT_DOCS
        pipe = DecoupledRAGPipeline(
            chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
        )
        pipe.ingest_documents(docs_to_use)
        st.session_state.pipeline = pipe
        st.session_state.synthetic_tests = SyntheticTestsetGenerator.generate_testset(
            pipe.chunks, num_questions=5
        )
        st.success(f"Indexed {len(pipe.chunks)} chunks & generated {len(st.session_state.synthetic_tests)} test cases.")

    st.divider()
    st.subheader("🧪 Component Fault Simulation")
    st.caption("Isolate Retriever vs. Generator failures to see the diagnostic engine in action:")
    scenario = st.radio(
        "Execution Scenario",
        options=[
            "✅ Normal End-to-End (Healthy)",
            "🔴 Retriever Failure (Wrong Chunks)",
            "🔴 Generator Hallucination (Unfaithful)",
            "🟠 Generator Omission (Missed Facts)",
            "🔴 Compound Failure (Both Fail)",
        ],
        index=0,
    )

    st.divider()
    openai_api_key = st.text_input(
        "OpenAI API Key (Optional)",
        type="password",
        help="Leave blank to use the built-in Local Semantic/Lexical Judge & Generator.",
    )
    pass_threshold = st.slider("Pass Threshold", 0.40, 0.90, 0.65, 0.05)
    st.caption(
        "Local mode uses BM25 + TF-IDF + regex numeric checks. Add a key to use the optional LLM judge."
    )

# Map scenario selection to fault flags
sim_bad_ret = scenario in ("🔴 Retriever Failure (Wrong Chunks)", "🔴 Compound Failure (Both Fail)")
if scenario in ("🔴 Generator Hallucination (Unfaithful)", "🔴 Compound Failure (Both Fail)"):
    sim_gen_mode = "hallucinate"
elif scenario == "🟠 Generator Omission (Missed Facts)":
    sim_gen_mode = "omit"
else:
    sim_gen_mode = "normal"

# ─── MAIN SINGLE-SCREEN DASHBOARD ───
st.title("🔬 RAG Evaluation Pipeline — Retriever & Generator Tested Apart")

# Top Query Selector Bar (Synthetic Test Questions or Custom Query)
test_options = ["[Custom Query]"] + [
    f"{tc.id} ({tc.question_type}): {tc.question}" for tc in st.session_state.synthetic_tests
]
col_q1, col_q2 = st.columns([3, 2])
with col_q1:
    selected_option = st.selectbox(
        "Select Auto-Generated Synthetic Test Question (Deliverable 2) or Enter Custom Query:",
        options=test_options,
        index=1 if len(test_options) > 1 else 0,
    )

if selected_option == "[Custom Query]":
    default_q = "How many books can a student borrow, and how long is the standard borrowing period?"
    default_gt = "Students may borrow up to 5 books at one time, and the standard borrowing period is 14 days."
else:
    idx = test_options.index(selected_option) - 1
    chosen_tc = st.session_state.synthetic_tests[idx]
    default_q = chosen_tc.question
    default_gt = chosen_tc.ground_truth

with col_q1:
    query_input = st.text_input("Query", value=default_q)
with col_q2:
    gt_input = st.text_area(
        "Ground Truth Reference (Optional — enables Context Recall & Answer Accuracy)",
        value=default_gt,
        height=92,
    )

# Run Pipeline & Evaluate
pipeline: DecoupledRAGPipeline = st.session_state.pipeline
evaluator = RAGEvaluator(threshold=pass_threshold)

retrieved_chunks = pipeline.retrieve(
    query=query_input, top_k=int(top_k), simulate_bad_retrieval=sim_bad_ret
)
generated_answer = pipeline.generate(
    query=query_input,
    retrieved_chunks=retrieved_chunks,
    openai_api_key=openai_api_key or None,
    simulate_mode=sim_gen_mode,
)
eval_result: EvaluationResult = evaluator.evaluate(
    query=query_input,
    retrieved_chunks=retrieved_chunks,
    generated_answer=generated_answer,
    ground_truth=gt_input.strip() if gt_input.strip() else None,
    openai_api_key=openai_api_key or None,
)

# ─── ROW 1: COLOUR-CODED SCORECARDS (Deliverable 1 & 4) ───
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    render_scorecard("1. Context Relevance", eval_result.context_relevance, "Retriever (Query ↔ Chunks)", pass_threshold)
with sc2:
    render_scorecard("2. Context Recall", eval_result.context_recall, "Retriever (GT ↔ Chunks)", pass_threshold)
with sc3:
    render_scorecard("3. Faithfulness", eval_result.faithfulness, "Generator (Answer ↔ Chunks)", pass_threshold)
with sc4:
    render_scorecard("4. Answer Accuracy", eval_result.answer_accuracy, "End-to-End (Answer ↔ GT)", pass_threshold)

# ─── ROW 2: AUTOMATED ROOT-CAUSE BANNER (Deliverable 3) ───
rc = eval_result.root_cause
banner_class = f"verdict-{rc.severity}"
st.markdown(
    f"""
    <div class="verdict-banner {banner_class}">
        <strong>🧭 Root-Cause Diagnosis: [{rc.verdict_code}] {rc.verdict_label}</strong>
        &nbsp;|&nbsp; <strong>Blamed Stage:</strong> <code>{rc.failed_component}</code><br/>
        {rc.summary}
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── ROW 3: SIDE-BY-SIDE INSPECTOR (Retrieved Context vs Generated Answer + Diagnostics) ───
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("📑 Stage 1: Retrieved Chunks (Retriever Output)")
    st.caption("Green = Relevant to query / contains ground truth · Red = Irrelevant chunk fetched")
    for ce in eval_result.chunk_evaluations:
        css_box = "chunk-green" if ce.is_relevant else "chunk-red"
        badge = "🟢 RELEVANT" if ce.is_relevant else "🔴 IRRELEVANT"
        gt_badge = " · 🎯 Contains Ground Truth" if ce.contains_ground_truth else ""
        st.markdown(
            f"""
            <div class="chunk-box {css_box}">
                <strong>[{html.escape(ce.chunk_id)}]</strong> ({badge} · Score: <code>{ce.relevance_score:.2f}</code>{gt_badge})<br/>
                {html.escape(ce.text)}
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.subheader("🤖 Stage 2: Generated Answer & Claim Verification")
    st.caption("Inline highlights: Green = Verified in Retrieved Context · Red = Hallucinated Claim")

    # Render inline color-coded answer
    highlighted_spans = []
    for cv in eval_result.claim_verifications:
        if cv.status == "SUPPORTED":
            span_cls = "claim-green"
            tag = f"✓ [{cv.supporting_chunk_id}]"
        elif cv.status == "HALLUCINATED":
            span_cls = "claim-red"
            tag = "✗ HALLUCINATED"
        else:
            span_cls = "claim-neutral"
            tag = "⚠ ABSTAINED"
        highlighted_spans.append(
            f'<span class="{span_cls}">{html.escape(cv.claim)} <strong>({html.escape(tag)})</strong></span>'
        )

    st.markdown(
        "<div style='line-height: 1.9; font-size: 0.97rem; margin-bottom: 12px;'>"
        + " ".join(highlighted_spans)
        + "</div>",
        unsafe_allow_html=True,
    )

    if eval_result.missing_ground_truth_facts:
        st.markdown(
            "<div class='chunk-box chunk-red'><strong>🔴 Missing Information (Present in Ground Truth but Omitted in Answer):</strong><br/>"
            + "<br/>".join(
                f"• {html.escape(m)}" for m in eval_result.missing_ground_truth_facts
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with st.expander("🛠️ Automated Root-Cause Remediation Plan & Claim Breakdown", expanded=True):
        st.markdown("**Prescribed Engineering Fixes:**")
        for step in rc.remediation_steps:
            st.markdown(f"- {step}")
        st.markdown("**Sentence-Level Grounding Audit:**")
        for cv in eval_result.claim_verifications:
            icon = "🟢" if cv.status == "SUPPORTED" else ("🔴" if cv.status == "HALLUCINATED" else "🟡")
            st.markdown(f"- {icon} **{cv.status}**: _{cv.claim}_ — {cv.reason}")

# ─── ROW 4: BATCH SYNTHETIC TESTSET RUNNER (Deliverable 2) ───
with st.expander("📊 View All Synthetic Test Questions & Run Full Batch Evaluation"):
    if st.button("▶️ Run Full Batch Evaluation Across All Synthetic Questions"):
        rows = []
        for tc in st.session_state.synthetic_tests:
            r_chunks = pipeline.retrieve(tc.question, top_k=int(top_k), simulate_bad_retrieval=sim_bad_ret)
            ans = pipeline.generate(
                tc.question, r_chunks, openai_api_key=openai_api_key or None, simulate_mode=sim_gen_mode
            )
            ev = evaluator.evaluate(
                tc.question, r_chunks, ans, tc.ground_truth, openai_api_key=openai_api_key or None
            )
            rows.append(
                {
                    "Test ID": tc.id,
                    "Type": tc.question_type,
                    "Question": tc.question,
                    "Context Relevance": f"{ev.context_relevance:.0%}",
                    "Context Recall": f"{ev.context_recall:.0%}" if ev.context_recall is not None else "N/A",
                    "Faithfulness": f"{ev.faithfulness:.0%}",
                    "Answer Accuracy": f"{ev.answer_accuracy:.0%}" if ev.answer_accuracy is not None else "N/A",
                    "Root-Cause Diagnosis": ev.root_cause.verdict_code,
                    "Blamed Stage": ev.root_cause.failed_component,
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
