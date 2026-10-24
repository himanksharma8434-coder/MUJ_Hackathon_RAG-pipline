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
    page_title="Campus RAG Lab",
    page_icon="🔍",
    layout="wide",
)

# Minimal black-and-white glassmorphism theme.
st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(255,255,255,.98), transparent 35%),
            linear-gradient(145deg, #f5f5f7 0%, #e9e9ed 100%);
        color: #111113;
    }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stHeader"] { background: rgba(255,255,255,0); }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1440px; padding: 2.4rem 3rem 4rem; }
    [data-testid="stSidebar"] {
        background: rgba(247,247,249,.82);
        border-right: 1px solid rgba(17,17,19,.08);
        backdrop-filter: blur(22px);
    }
    [data-testid="stSidebarContent"] { padding: 1.4rem 1rem 2rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(17,17,19,.10);
        border-radius: 24px;
        background: rgba(255,255,255,.58);
        box-shadow: 0 18px 50px rgba(17,17,19,.07);
        backdrop-filter: blur(20px);
    }
    h1, h2, h3 { letter-spacing: -0.045em; color: #111113; }
    h1 { font-size: clamp(2.2rem, 5vw, 4.4rem) !important; line-height: .98 !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.05rem !important; }
    p, label, .stCaption { color: #6b6b73; }
    .hero {
        padding: 2.2rem 2.4rem 2rem;
        border: 1px solid rgba(17,17,19,.10);
        border-radius: 30px;
        background: linear-gradient(135deg, rgba(255,255,255,.88), rgba(255,255,255,.42));
        box-shadow: 0 24px 70px rgba(17,17,19,.10);
        backdrop-filter: blur(24px);
        margin-bottom: 1.2rem;
    }
    .eyebrow { color: #707078; font-size: .72rem; font-weight: 700; letter-spacing: .16em; text-transform: uppercase; }
    .hero-title { margin: .45rem 0 .7rem; font-size: clamp(2.2rem, 5vw, 4.2rem); font-weight: 800; line-height: .98; letter-spacing: -.07em; }
    .hero-copy { max-width: 680px; margin: 0; font-size: 1rem; color: #6b6b73; }
    .micro-status { margin-top: 1.2rem; color: #6b6b73; font-size: .82rem; }
    .micro-status strong { color: #111113; }
    .metric-card {
        min-height: 118px;
        padding: 1.1rem 1.15rem;
        border: 1px solid rgba(17,17,19,.10);
        border-radius: 20px;
        background: rgba(255,255,255,.64);
        box-shadow: 0 12px 35px rgba(17,17,19,.06);
        backdrop-filter: blur(18px);
    }
    .metric-top { display: flex; align-items: center; justify-content: space-between; gap: .5rem; color: #6b6b73; font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .metric-dot { width: 8px; height: 8px; border-radius: 50%; background: #111113; display: inline-block; }
    .metric-dot.fail { background: #b5b5bd; box-shadow: 0 0 0 4px rgba(17,17,19,.08); }
    .metric-value { margin: .3rem 0 .05rem; color: #111113; font-size: 2rem; font-weight: 800; letter-spacing: -.06em; }
    .metric-sub { color: #87878e; font-size: .74rem; }
    .verdict-banner { padding: 1.1rem 1.25rem; border: 1px solid #111113; border-radius: 18px; background: rgba(255,255,255,.74); box-shadow: 0 12px 30px rgba(17,17,19,.06); }
    .verdict-banner.warning { border-color: #8b8b92; }
    .verdict-banner.error { border-color: #111113; background: rgba(241,241,243,.88); }
    .verdict-kicker { color: #74747c; font-size: .7rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .verdict-title { margin: .25rem 0 .35rem; color: #111113; font-size: 1.25rem; font-weight: 800; letter-spacing: -.04em; }
    .verdict-copy { margin: 0; color: #6b6b73; font-size: .88rem; line-height: 1.45; }
    .answer-card { padding: 1.25rem 1.4rem; border: 1px solid rgba(17,17,19,.10); border-radius: 22px; background: rgba(255,255,255,.74); box-shadow: 0 15px 40px rgba(17,17,19,.06); }
    .answer-label { color: #74747c; font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .answer-text { margin-top: .7rem; color: #202024; font-size: 1.08rem; line-height: 1.7; }
    .chunk-box { border-radius: 14px; padding: 12px 14px; margin-bottom: 9px; font-size: .86rem; line-height: 1.5; background: rgba(255,255,255,.76); border: 1px solid rgba(17,17,19,.10); color: #25252a; }
    .chunk-green { border-left: 4px solid #111113; }
    .chunk-red { border-left: 4px solid #a4a4ab; background: rgba(238,238,241,.78); }
    .claim-green { background: #f0f0f2; color: #111113; padding: 3px 7px; border-radius: 6px; border-bottom: 2px solid #111113; }
    .claim-red { background: #dedee2; color: #111113; padding: 3px 7px; border-radius: 6px; border-bottom: 2px solid #77777f; font-weight: 700; }
    .claim-neutral { background: #f5f5f7; color: #55555d; padding: 3px 7px; border-radius: 6px; border-bottom: 2px solid #aaaab1; }
    .section-label { margin: 1.5rem 0 .6rem; color: #74747c; font-size: .72rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
    .sidebar-brand { margin-bottom: 1.3rem; color: #111113; font-size: 1.15rem; font-weight: 800; letter-spacing: -.04em; }
    .sidebar-note { margin: .2rem 0 1.2rem; color: #7a7a82; font-size: .78rem; line-height: 1.45; }
    .stButton > button { border: 1px solid #111113; border-radius: 14px; background: #111113; color: white; font-weight: 700; min-height: 2.6rem; box-shadow: 0 8px 18px rgba(17,17,19,.16); }
    .stButton > button:hover { border-color: #333338; background: #333338; color: white; }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] > div { border-radius: 14px; border-color: rgba(17,17,19,.15); background: rgba(255,255,255,.74); }
    .stExpander { border-color: rgba(17,17,19,.10) !important; border-radius: 16px !important; background: rgba(255,255,255,.34); }
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
        val_str = "N/A"
        status_str = "No Ground Truth"
        dot_class = "fail"
    else:
        is_good = score >= threshold
        val_str = f"{score:.0%}"
        status_str = "On target" if is_good else "Below target"
        dot_class = "" if is_good else "fail"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-top"><span>{title}</span><span class="metric-dot {dot_class}"></span></div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-sub">{status_str} · {subtitle}</div>
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
    st.markdown('<div class="sidebar-brand">Campus RAG Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-note">A simple way to show whether a wrong answer came from search or generation.</div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Add documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", use_container_width=True):
        docs_to_use = parse_uploaded_files(uploaded_files) if uploaded_files else DEFAULT_DOCS
        pipe = DecoupledRAGPipeline(
            chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
        )
        pipe.ingest_documents(docs_to_use)
        st.session_state.pipeline = pipe
        st.session_state.synthetic_tests = SyntheticTestsetGenerator.generate_testset(
            pipe.chunks, num_questions=5
        )
        st.success(f"Ready · {len(pipe.chunks)} chunks")

    st.markdown('<div class="section-label">Demo mode</div>', unsafe_allow_html=True)
    scenario = st.radio(
        "Choose one scenario",
        options=[
            "Normal · healthy",
            "Retriever · wrong chunks",
            "Generator · hallucination",
            "Generator · omission",
            "Compound · both fail",
        ],
        index=0,
    )

    with st.expander("Advanced settings"):
        col_c1, col_c2 = st.columns(2)
        chunk_size = col_c1.slider("Chunk size", 30, 300, 65, step=5)
        top_k = col_c2.slider("Top-K", 1, 6, 2)
        overlap_max = max(0, min(100, int(chunk_size) - 1))
        chunk_overlap = st.slider(
            "Chunk overlap", 0, overlap_max, min(12, overlap_max), step=1
        )
        pass_threshold = st.slider("Pass threshold", 0.40, 0.90, 0.65, 0.05)
        openai_api_key = st.text_input(
            "LLM judge key (optional)",
            type="password",
            help="Leave blank for the local zero-key evaluator.",
        )
        st.caption("Local mode: BM25 + TF-IDF + numeric consistency checks.")

# Map scenario selection to fault flags
sim_bad_ret = scenario in ("Retriever · wrong chunks", "Compound · both fail")
if scenario in ("Generator · hallucination", "Compound · both fail"):
    sim_gen_mode = "hallucinate"
elif scenario == "Generator · omission":
    sim_gen_mode = "omit"
else:
    sim_gen_mode = "normal"

# ─── MAIN SINGLE-SCREEN DASHBOARD ───
st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">RAG / ROOT-CAUSE / LIVE DEMO</div>
        <div class="hero-title">Can we trust<br/>this answer?</div>
        <p class="hero-copy">One clean view separates a search problem from a generation problem.</p>
        <div class="micro-status"><strong>Ask</strong> → read the four scores → open evidence only when needed.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Query Selector Bar (Synthetic Test Questions or Custom Query)
test_options = ["[Custom Query]"] + [
    f"{tc.id} ({tc.question_type}): {tc.question}" for tc in st.session_state.synthetic_tests
]
with st.container(border=True):
    st.markdown('<div class="section-label">1 · Ask a question</div>', unsafe_allow_html=True)
    selected_option = st.selectbox(
        "Question library",
        options=test_options,
        index=1 if len(test_options) > 1 else 0,
        label_visibility="collapsed",
    )

if selected_option == "[Custom Query]":
    default_q = "How many books can a student borrow, and how long is the standard borrowing period?"
    default_gt = "Students may borrow up to 5 books at one time, and the standard borrowing period is 14 days."
else:
    idx = test_options.index(selected_option) - 1
    chosen_tc = st.session_state.synthetic_tests[idx]
    default_q = chosen_tc.question
    default_gt = chosen_tc.ground_truth

query_input = st.text_input("Your question", value=default_q)
with st.expander("Edit reference answer (optional)"):
    gt_input = st.text_area(
        "Reference answer",
        value=default_gt,
        height=78,
        help="Used to measure context recall and answer accuracy.",
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

# ─── FAST READ: four scores + one verdict ───
st.markdown('<div class="section-label">2 · Read the result</div>', unsafe_allow_html=True)
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    render_scorecard("Relevance", eval_result.context_relevance, "search", pass_threshold)
with sc2:
    render_scorecard("Recall", eval_result.context_recall, "search", pass_threshold)
with sc3:
    render_scorecard("Faithfulness", eval_result.faithfulness, "answer", pass_threshold)
with sc4:
    render_scorecard("Accuracy", eval_result.answer_accuracy, "answer", pass_threshold)

rc = eval_result.root_cause
st.markdown(
    f"""
    <div class="verdict-banner {html.escape(rc.severity)}">
        <div class="verdict-kicker">Root cause · {html.escape(rc.failed_component)}</div>
        <div class="verdict-title">{html.escape(rc.verdict_code)} — {html.escape(rc.verdict_label)}</div>
        <p class="verdict-copy">{html.escape(rc.summary)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

highlighted_spans = []
for cv in eval_result.claim_verifications:
    if cv.status == "SUPPORTED":
        span_cls = "claim-green"
        tag = f"✓ {cv.supporting_chunk_id or 'context'}"
    elif cv.status == "HALLUCINATED":
        span_cls = "claim-red"
        tag = "✗ unsupported"
    else:
        span_cls = "claim-neutral"
        tag = "⚠ abstained"
    highlighted_spans.append(
        f'<span class="{span_cls}">{html.escape(cv.claim)} <strong>({html.escape(tag)})</strong></span>'
    )

st.markdown(
    '<div class="answer-card">'
    '<div class="answer-label">3 · Final answer</div>'
    '<div class="answer-text">'
    + (" ".join(highlighted_spans) or html.escape(generated_answer))
    + "</div></div>",
    unsafe_allow_html=True,
)

with st.expander("See evidence · retrieved context and claim audit"):
    left_col, right_col = st.columns([1, 1], gap="large")
    with left_col:
        st.markdown("**Retrieved context**")
        st.caption("The chunks the answer was allowed to use.")
        for ce in eval_result.chunk_evaluations:
            css_box = "chunk-green" if ce.is_relevant else "chunk-red"
            badge = "RELEVANT" if ce.is_relevant else "IRRELEVANT"
            gt_badge = " · ground truth" if ce.contains_ground_truth else ""
            st.markdown(
                f"""
                <div class="chunk-box {css_box}">
                    <strong>{html.escape(ce.chunk_id)}</strong> · {badge} · {ce.relevance_score:.2f}{gt_badge}<br/>
                    {html.escape(ce.text)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right_col:
        st.markdown("**Claim audit**")
        for cv in eval_result.claim_verifications:
            icon = "✓" if cv.status == "SUPPORTED" else ("✗" if cv.status == "HALLUCINATED" else "⚠")
            st.markdown(
                f"{icon} **{cv.status.title()}** — {html.escape(cv.claim)}  \n"
                f"<small>{html.escape(cv.reason)}</small>",
                unsafe_allow_html=True,
            )
        if eval_result.missing_ground_truth_facts:
            st.markdown("**Missing facts**")
            for missing in eval_result.missing_ground_truth_facts:
                st.markdown(f"- {html.escape(missing)}")
        st.markdown("**Recommended next step**")
        for step in rc.remediation_steps:
            st.markdown(f"- {html.escape(step)}")

# ─── OPTIONAL DEEP DIVE: BATCH SYNTHETIC TESTSET RUNNER ───
with st.expander("Run batch audit · optional"):
    st.caption("Evaluate every generated question with the same scenario.")
    if st.button("Run batch audit"):
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
