"""Campus RAG Lab Streamlit application.

The app has three intentionally different views:

* Overview: the five-second scorecard for a demo audience.
* User ↔ AI: a clean, student-facing chat experience.
* AI Engineer: the detailed root-cause and remediation view.
"""

from __future__ import annotations

import io
from typing import Optional

import streamlit as st
from pypdf import PdfReader

from evaluator import EvaluationResult, RAGEvaluator
from rag_pipeline import DecoupledRAGPipeline
from test_set_generator import SyntheticTestsetGenerator
from ui_views import (
    render_batch_audit,
    render_chat,
    render_engineer,
    render_overview,
)


st.set_page_config(
    page_title="Campus RAG Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(255,255,255,.98), transparent 34%),
            linear-gradient(145deg, #f8f8f6 0%, #e8e8e5 100%);
        color: #111113;
    }
    [data-testid="stAppViewContainer"] { background: transparent; }
    [data-testid="stHeader"] { background: rgba(255,255,255,0); }
    [data-testid="stToolbar"] { visibility: hidden; }
    .block-container { max-width: 1440px; padding: 2.2rem 3rem 4rem; }
    [data-testid="stSidebar"] {
        background: rgba(9,10,13,.97);
        border-right: 1px solid rgba(255,255,255,.12);
        backdrop-filter: blur(22px);
    }
    [data-testid="stSidebarContent"] { padding: 1.2rem 1rem 2rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 24px;
        background: linear-gradient(145deg, rgba(16,17,21,.98), rgba(7,8,11,.94));
        box-shadow: 0 22px 60px rgba(17,17,19,.19), inset 0 1px 0 rgba(255,255,255,.09);
        backdrop-filter: blur(20px);
        color: #fff;
    }
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] label,
    [data-testid="stVerticalBlockBorderWrapper"] .stCaption { color: rgba(255,255,255,.68) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] .section-label { color: rgba(255,255,255,.55) !important; }
    h1, h2, h3 { letter-spacing: -.045em; color: #111113; }
    h1 { font-size: clamp(2.2rem, 5vw, 4.4rem) !important; line-height: .98 !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.05rem !important; }
    p, label, .stCaption { color: #6b6b73; }
    .page-header {
        padding: 1.2rem .15rem 1.1rem;
        animation: rise-in .55s cubic-bezier(.2,.75,.2,1) both;
    }
    .eyebrow, .section-label, .verdict-kicker, .answer-label {
        color: #74747c;
        font-size: .7rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .page-title {
        margin: .35rem 0 .45rem;
        color: #111113;
        font-size: clamp(2.2rem, 5vw, 4.2rem);
        font-weight: 850;
        line-height: .96;
        letter-spacing: -.07em;
    }
    .page-copy { max-width: 720px; margin: 0; font-size: 1rem; color: #6b6b73; }
    .section-label { margin: 1.45rem 0 .6rem; }
    .metric-card {
        min-height: 118px;
        padding: 1.1rem 1.15rem;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 20px;
        background: linear-gradient(145deg, rgba(17,18,22,.98), rgba(8,9,12,.94));
        box-shadow: 0 16px 42px rgba(17,17,19,.18), inset 0 1px 0 rgba(255,255,255,.10);
        backdrop-filter: blur(18px);
    }
    .metric-top { display: flex; align-items: center; justify-content: space-between; color: rgba(255,255,255,.62); font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
    .metric-dot { width: 8px; height: 8px; border-radius: 50%; background: #fff; display: inline-block; box-shadow: 0 0 14px rgba(255,255,255,.55); }
    .metric-dot.fail { background: #787b84; box-shadow: 0 0 0 4px rgba(255,255,255,.08); }
    .metric-value { margin: .3rem 0 .05rem; color: #fff; font-size: 2rem; font-weight: 800; letter-spacing: -.06em; }
    .metric-sub { color: rgba(255,255,255,.48); font-size: .74rem; }
    .verdict-banner {
        margin-top: .75rem;
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(17,18,22,.98), rgba(8,9,12,.94));
        box-shadow: 0 16px 42px rgba(17,17,19,.18), inset 0 1px 0 rgba(255,255,255,.10);
        animation: rise-in .5s .08s cubic-bezier(.2,.75,.2,1) both;
    }
    .verdict-banner.warning { border-color: rgba(255,255,255,.42); }
    .verdict-banner.error { background: linear-gradient(145deg, rgba(28,29,34,.98), rgba(10,11,14,.96)); }
    .verdict-title { margin: .25rem 0 .35rem; color: #fff; font-size: 1.25rem; font-weight: 800; letter-spacing: -.04em; }
    .verdict-copy { margin: 0; color: rgba(255,255,255,.62); font-size: .88rem; line-height: 1.45; }
    .answer-card {
        margin-top: .8rem;
        padding: 1.25rem 1.4rem;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 22px;
        background: linear-gradient(145deg, rgba(17,18,22,.98), rgba(8,9,12,.94));
        box-shadow: 0 16px 42px rgba(17,17,19,.18), inset 0 1px 0 rgba(255,255,255,.10);
        animation: rise-in .55s .14s cubic-bezier(.2,.75,.2,1) both;
    }
    .answer-text { margin-top: .7rem; color: #fff; font-size: 1.08rem; line-height: 1.7; }
    .claim-green { background: rgba(255,255,255,.10); color: #fff; padding: 3px 7px; border-radius: 6px; border: 1px solid rgba(255,255,255,.26); }
    .claim-red { background: rgba(255,255,255,.16); color: #fff; padding: 3px 7px; border-radius: 6px; border: 1px solid rgba(255,255,255,.42); font-weight: 700; }
    .claim-neutral { background: rgba(255,255,255,.06); color: rgba(255,255,255,.72); padding: 3px 7px; border-radius: 6px; border: 1px solid rgba(255,255,255,.18); }
    .chunk-box { border-radius: 14px; padding: 12px 14px; margin-bottom: 9px; font-size: .86rem; line-height: 1.5; background: linear-gradient(145deg, rgba(18,19,23,.98), rgba(8,9,12,.94)); border: 1px solid rgba(255,255,255,.14); color: rgba(255,255,255,.82); box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }
    .chunk-green { border-left: 4px solid #fff; }
    .chunk-red { border-left: 4px solid #71747d; background: linear-gradient(145deg, rgba(34,35,41,.98), rgba(13,14,18,.96)); }
    .sidebar-brand { margin-bottom: .25rem; color: #fff; font-size: 1.15rem; font-weight: 800; letter-spacing: -.04em; }
    .sidebar-note { margin: .2rem 0 1.2rem; color: rgba(255,255,255,.54); font-size: .78rem; line-height: 1.45; }
    .stButton > button {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.28);
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(35,35,42,.88), rgba(12,12,15,.76));
        color: #fff !important;
        font-weight: 700;
        min-height: 2.6rem;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 10px 24px rgba(17,17,19,.18);
        backdrop-filter: blur(18px);
        transition: transform .25s ease, box-shadow .25s ease, background .25s ease;
    }
    .stButton > button::before { content: ""; position: absolute; inset: 0; background: linear-gradient(115deg, transparent 20%, rgba(255,255,255,.22) 48%, transparent 75%); transform: translateX(-120%); transition: transform .55s ease; pointer-events: none; }
    .stButton > button:hover { transform: translateY(-2px); background: linear-gradient(135deg, rgba(50,50,58,.92), rgba(18,18,22,.82)); box-shadow: inset 0 1px 0 rgba(255,255,255,.22), 0 14px 30px rgba(17,17,19,.22); }
    .stButton > button:hover::before { transform: translateX(120%); }
    .stButton > button:active { transform: translateY(0) scale(.985); }
    [data-testid="stSidebar"] .section-label { color: rgba(255,255,255,.46); }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,.68) !important; }
    [data-testid="stSidebar"] [data-baseweb="radio"] label { color: rgba(255,255,255,.78) !important; }
    [data-testid="stFileUploaderDropzone"] { border: 1px solid rgba(255,255,255,.16) !important; border-radius: 18px !important; background: linear-gradient(145deg, rgba(28,29,34,.98), rgba(9,10,13,.95)) !important; box-shadow: inset 0 1px 0 rgba(255,255,255,.10), 0 12px 28px rgba(17,17,19,.15); backdrop-filter: blur(18px); color: rgba(255,255,255,.72); }
    [data-testid="stFileUploaderDropzone"] section { background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] button { border: 1px solid rgba(255,255,255,.26) !important; border-radius: 10px !important; background: rgba(255,255,255,.10) !important; color: #fff !important; }
    .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stSelectbox [data-baseweb="select"] > div {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,.16) !important;
        background: linear-gradient(145deg, rgba(22,23,28,.98), rgba(9,10,13,.95)) !important;
        color: #fff !important;
        box-shadow: inset 0 1px 0 rgba(255,255,255,.09), 0 8px 24px rgba(17,17,19,.15);
        backdrop-filter: blur(18px);
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder { color: rgba(255,255,255,.42) !important; }
    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox [data-baseweb="select"] > div:focus-within { border-color: rgba(255,255,255,.42) !important; box-shadow: 0 0 0 4px rgba(255,255,255,.08), inset 0 1px 0 rgba(255,255,255,.14) !important; }
    .stSelectbox [data-baseweb="select"] * { color: #fff !important; }
    [role="listbox"], [role="option"] { background: #111216 !important; color: #fff !important; }
    [data-baseweb="radio"] [role="radio"] { border-color: rgba(255,255,255,.62) !important; background: rgba(255,255,255,.08) !important; }
    [data-baseweb="radio"] [role="radio"][aria-checked="true"] { background: #fff !important; box-shadow: inset 0 0 0 3px #111216 !important; }
    [data-testid="stChatMessage"] { border: 1px solid rgba(255,255,255,.14); border-radius: 20px; margin: .65rem 0; padding: .85rem 1rem; background: linear-gradient(145deg, rgba(18,19,23,.98), rgba(8,9,12,.94)); color: #fff; box-shadow: 0 14px 34px rgba(17,17,19,.17), inset 0 1px 0 rgba(255,255,255,.09); backdrop-filter: blur(18px); animation: rise-in .4s cubic-bezier(.2,.75,.2,1) both; }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] span { color: rgba(255,255,255,.86) !important; }
    [data-testid="stChatInput"] { border-radius: 18px !important; background: linear-gradient(145deg, rgba(18,19,23,.98), rgba(8,9,12,.94)) !important; box-shadow: 0 16px 36px rgba(17,17,19,.22), inset 0 1px 0 rgba(255,255,255,.12); backdrop-filter: blur(20px); }
    [data-testid="stChatInput"] textarea { background: transparent !important; color: #fff !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: rgba(255,255,255,.44) !important; }
    .chat-meta { display: flex; gap: .55rem; align-items: center; margin-top: .7rem; color: rgba(255,255,255,.46); font-size: .74rem; }
    .chat-pill { padding: .25rem .55rem; border: 1px solid rgba(255,255,255,.20); border-radius: 999px; background: rgba(255,255,255,.08); color: rgba(255,255,255,.78); }
    .active-run { display: flex; flex-direction: column; gap: .25rem; margin: .4rem 0 1rem; padding: .85rem 1rem; border: 1px solid rgba(255,255,255,.14); border-radius: 16px; background: linear-gradient(145deg, rgba(18,19,23,.98), rgba(8,9,12,.94)); color: rgba(255,255,255,.54); font-size: .74rem; box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }
    .active-run strong { color: #fff; font-size: .92rem; font-weight: 650; }
    .problem-row, .remediation-row { margin: .45rem 0; padding: .8rem .95rem; border: 1px solid rgba(255,255,255,.14); border-radius: 14px; background: linear-gradient(145deg, rgba(18,19,23,.98), rgba(8,9,12,.94)); color: rgba(255,255,255,.82); font-size: .88rem; line-height: 1.45; box-shadow: inset 0 1px 0 rgba(255,255,255,.08); }
    .problem-row { border-left: 4px solid #fff; }
    .remediation-row { border-left: 4px solid #777b84; }
    .stExpander { border-color: rgba(255,255,255,.14) !important; border-radius: 16px !important; background: linear-gradient(145deg, rgba(18,19,23,.96), rgba(8,9,12,.90)); color: #fff; }
    .stExpander summary, .stExpander summary p { color: rgba(255,255,255,.82) !important; }
    .fade-in { animation: rise-in .5s cubic-bezier(.2,.75,.2,1) both; }
    @keyframes rise-in { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


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


SCENARIOS = [
    "Normal · healthy",
    "Retriever · wrong chunks",
    "Generator · hallucination",
    "Generator · omission",
    "Compound · both fail",
]


def parse_uploaded_files(uploaded_files) -> list[dict]:
    documents = []
    for uploaded_file in uploaded_files or []:
        try:
            if uploaded_file.name.lower().endswith(".pdf"):
                reader = PdfReader(io.BytesIO(uploaded_file.read()))
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            else:
                text = uploaded_file.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            st.warning(f"Could not read {uploaded_file.name}: {exc}")
            continue
        if text.strip():
            documents.append({"name": uploaded_file.name, "content": text})
    return documents


def ensure_session_state() -> None:
    if "pipeline" not in st.session_state:
        pipeline = DecoupledRAGPipeline(chunk_size=65, chunk_overlap=12)
        pipeline.ingest_documents(DEFAULT_DOCS)
        st.session_state.pipeline = pipeline
        st.session_state.synthetic_tests = SyntheticTestsetGenerator.generate_testset(
            pipeline.chunks, num_questions=5
        )
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("latest_query", "How many books can a student borrow, and how long is the standard borrowing period?")
    st.session_state.setdefault("latest_ground_truth", "Students may borrow up to 5 books at one time, and the standard borrowing period is 14 days.")


def render_sidebar() -> tuple[str, str, int, float, Optional[str], int, bool, str]:
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">Campus RAG Lab</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-note">A simple way to show whether a wrong answer came from search or generation.</div>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Workspace",
            ["Overview", "User ↔ AI", "AI engineer"],
            label_visibility="collapsed",
        )
        st.markdown('<div class="section-label">Documents</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Add documents",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        if st.button("Index documents", use_container_width=True):
            documents = parse_uploaded_files(uploaded_files) if uploaded_files else DEFAULT_DOCS
            pipeline = DecoupledRAGPipeline(chunk_size=65, chunk_overlap=12)
            pipeline.ingest_documents(documents)
            st.session_state.pipeline = pipeline
            st.session_state.synthetic_tests = SyntheticTestsetGenerator.generate_testset(
                pipeline.chunks, num_questions=5
            )
            st.session_state.chat_history = []
            st.success(f"Ready · {len(pipeline.chunks)} chunks")

        st.markdown('<div class="section-label">Demo mode</div>', unsafe_allow_html=True)
        scenario = st.radio("Choose one scenario", SCENARIOS, index=0)

        with st.expander("Advanced settings"):
            col_one, col_two = st.columns(2)
            chunk_size = col_one.slider("Chunk size", 30, 300, 65, step=5)
            top_k = col_two.slider("Top-K", 1, 6, 2)
            overlap_max = max(0, min(100, int(chunk_size) - 1))
            chunk_overlap = st.slider("Chunk overlap", 0, overlap_max, min(12, overlap_max), step=1)
            pass_threshold = st.slider("Pass threshold", 0.40, 0.90, 0.65, 0.05)
            openai_api_key = st.text_input(
                "LLM judge key (optional)",
                type="password",
                help="Leave blank for the local zero-key evaluator.",
            )
            st.caption("Local mode: BM25 + TF-IDF + numeric consistency checks.")

    simulate_bad_retrieval = scenario in ("Retriever · wrong chunks", "Compound · both fail")
    if scenario in ("Generator · hallucination", "Compound · both fail"):
        simulate_mode = "hallucinate"
    elif scenario == "Generator · omission":
        simulate_mode = "omit"
    else:
        simulate_mode = "normal"
    return (
        page,
        scenario,
        int(top_k),
        float(pass_threshold),
        openai_api_key or None,
        int(chunk_overlap),
        simulate_bad_retrieval,
        simulate_mode,
    )


def main() -> None:
    ensure_session_state()
    (
        page,
        _scenario,
        top_k,
        pass_threshold,
        openai_api_key,
        _chunk_overlap,
        simulate_bad_retrieval,
        simulate_mode,
    ) = render_sidebar()

    pipeline: DecoupledRAGPipeline = st.session_state.pipeline
    evaluator = RAGEvaluator(threshold=pass_threshold)

    def run_evaluation(query: str, ground_truth: Optional[str]) -> EvaluationResult:
        chunks = pipeline.retrieve(
            query=query,
            top_k=top_k,
            simulate_bad_retrieval=simulate_bad_retrieval,
        )
        answer = pipeline.generate(
            query=query,
            retrieved_chunks=chunks,
            openai_api_key=openai_api_key,
            simulate_mode=simulate_mode,
        )
        return evaluator.evaluate(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
            ground_truth=ground_truth,
            openai_api_key=openai_api_key,
        )

    if page == "Overview":
        render_overview(
            pipeline,
            evaluator,
            st.session_state.synthetic_tests,
            top_k,
            simulate_bad_retrieval,
            simulate_mode,
            openai_api_key,
            run_evaluation,
            pass_threshold,
        )
        render_batch_audit(
            pipeline,
            evaluator,
            st.session_state.synthetic_tests,
            top_k,
            simulate_bad_retrieval,
            simulate_mode,
            openai_api_key,
        )
    elif page == "User ↔ AI":
        render_chat(st.session_state.synthetic_tests, run_evaluation, pass_threshold)
    else:
        current_query = st.session_state.get("latest_query")
        current_ground_truth = st.session_state.get("latest_ground_truth")
        if st.session_state.get("chat_history"):
            current_query = st.session_state.chat_history[-1]["query"]
            current_ground_truth = st.session_state.chat_history[-1].get("ground_truth")
        render_engineer(current_query, current_ground_truth, run_evaluation, pass_threshold)
        render_batch_audit(
            pipeline,
            evaluator,
            st.session_state.synthetic_tests,
            top_k,
            simulate_bad_retrieval,
            simulate_mode,
            openai_api_key,
        )


main()
