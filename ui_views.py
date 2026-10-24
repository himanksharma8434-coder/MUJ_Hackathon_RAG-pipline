"""Presentation views for the Campus RAG Lab Streamlit application."""

from __future__ import annotations

import html
from typing import Callable, Iterable, Optional

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as components_html

from evaluator import EvaluationResult, RAGEvaluator
from rag_pipeline import DecoupledRAGPipeline
from test_set_generator import SyntheticTestCase


EvaluationRunner = Callable[[str, Optional[str]], EvaluationResult]


def _escape(value: object) -> str:
    return html.escape(str(value))


def _render_html(markup: str, height: int) -> None:
    """Use the native HTML element when available, with an older Streamlit fallback."""

    if hasattr(st, "html"):
        st.html(markup)
    else:
        components_html(markup, height=height, scrolling=False)


def render_page_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="page-header fade-in">
            <div class="eyebrow">{_escape(kicker)}</div>
            <div class="page-title">{_escape(title)}</div>
            <p class="page-copy">{_escape(copy)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_animated_hero(kicker: str, title: str, copy: str) -> None:
    """Render a small GSAP-powered hero; the page remains usable if CDN loading is blocked."""

    _render_html(
        f"""
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8" />
            <script src="https://cdn.jsdelivr.net/npm/gsap@3.12.5/dist/gsap.min.js"></script>
            <style>
              * {{ box-sizing: border-box; }}
              html, body {{ margin: 0; width: 100%; height: 100%; overflow: hidden; background: transparent; }}
              body {{ padding-bottom: 12px; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
              .hero {{ position: relative; isolation: isolate; min-height: 214px; overflow: hidden; padding: 32px 38px 28px; border: 1px solid rgba(17,17,19,.10); border-radius: 30px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(255,255,255,.48)); box-shadow: 0 24px 70px rgba(17,17,19,.10), inset 0 1px 0 rgba(255,255,255,.8); backdrop-filter: blur(24px); }}
              .hero::after {{ content: ""; position: absolute; z-index: -1; width: 300px; height: 300px; right: -90px; top: -130px; border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.94) 0%, rgba(255,255,255,0) 70%); }}
              .orb {{ position: absolute; z-index: -1; width: 120px; height: 120px; right: 18%; bottom: -74px; border-radius: 50%; background: rgba(17,17,19,.045); filter: blur(2px); }}
              .eyebrow {{ color: #707078; font-size: 11px; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }}
              .hero-title {{ margin: 8px 0 10px; color: #111113; font-size: clamp(42px, 5vw, 68px); font-weight: 850; line-height: .92; letter-spacing: -.075em; }}
              .hero-copy {{ max-width: 680px; margin: 0; color: #6b6b73; font-size: 15px; }}
              .micro-status {{ margin-top: 18px; color: #6b6b73; font-size: 12px; }}
              .micro-status strong {{ color: #111113; }}
              .reveal {{ opacity: 1; transform: translateY(0); }}
              @media (max-width: 720px) {{ .hero {{ padding: 28px 24px; }} .hero-title {{ font-size: 48px; }} }}
            </style>
          </head>
          <body>
            <div class="hero">
              <div class="orb"></div>
              <div class="eyebrow reveal">{_escape(kicker)}</div>
              <div class="hero-title reveal">{_escape(title)}</div>
              <p class="hero-copy reveal">{_escape(copy)}</p>
              <div class="micro-status reveal"><strong>Ask</strong> → read the result → open evidence when needed.</div>
            </div>
            <script>
              const start = () => {{
                if (!window.gsap) return;
                gsap.set(".reveal", {{ opacity: 0, y: 16 }});
                gsap.timeline({{ defaults: {{ ease: "power3.out" }} }})
                  .to(".hero", {{ duration: .65, opacity: 1, y: 0, startAt: {{ opacity: 0, y: 10 }} }})
                  .to(".eyebrow", {{ duration: .36, opacity: 1, y: 0 }}, "-=.38")
                  .to(".hero-title", {{ duration: .56, opacity: 1, y: 0 }}, "-=.22")
                  .to(".hero-copy", {{ duration: .38, opacity: 1, y: 0 }}, "-=.28")
                  .to(".micro-status", {{ duration: .38, opacity: 1, y: 0 }}, "-=.22");
                gsap.to(".orb", {{ duration: 5, x: 28, y: -10, repeat: -1, yoyo: true, ease: "sine.inOut" }});
              }};
              window.addEventListener("load", start);
            </script>
          </body>
        </html>
        """,
        238,
    )


def render_scorecard(title: str, score: Optional[float], subtitle: str, threshold: float) -> None:
    if score is None:
        value = "N/A"
        status = "Needs reference"
        dot_class = "fail"
    else:
        value = f"{score:.0%}"
        passing = score >= threshold
        status = "On target" if passing else "Below target"
        dot_class = "" if passing else "fail"
    st.markdown(
        f"""
        <div class="metric-card fade-in">
            <div class="metric-top"><span>{_escape(title)}</span><span class="metric-dot {dot_class}"></span></div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{_escape(status)} · {_escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict(result: EvaluationResult) -> None:
    report = result.root_cause
    if report is None:
        return
    st.markdown(
        f"""
        <div class="verdict-banner { _escape(report.severity) } fade-in">
            <div class="verdict-kicker">Root cause · {_escape(report.failed_component)}</div>
            <div class="verdict-title">{_escape(report.verdict_code)} — {_escape(report.verdict_label)}</div>
            <p class="verdict-copy">{_escape(report.summary)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _claim_markup(result: EvaluationResult) -> str:
    spans = []
    for claim in result.claim_verifications:
        if claim.status == "SUPPORTED":
            css_class = "claim-green"
            tag = f"✓ {claim.supporting_chunk_id or 'context'}"
        elif claim.status == "HALLUCINATED":
            css_class = "claim-red"
            tag = "✗ unsupported"
        else:
            css_class = "claim-neutral"
            tag = "⚠ abstained"
        spans.append(
            f'<span class="{css_class}">{_escape(claim.claim)} '
            f'<strong>({_escape(tag)})</strong></span>'
        )
    return " ".join(spans) or _escape(result.generated_answer)


def render_answer_card(result: EvaluationResult, label: str = "Final answer") -> None:
    st.markdown(
        f"""
        <div class="answer-card fade-in">
            <div class="answer-label">{_escape(label)}</div>
            <div class="answer-text">{_claim_markup(result)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence(result: EvaluationResult, report_title: str = "Evidence") -> None:
    with st.expander(report_title):
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown("**Retrieved context**")
            st.caption("The chunks the answer was allowed to use.")
            if not result.chunk_evaluations:
                st.info("No context was retrieved.")
            for chunk in result.chunk_evaluations:
                css_class = "chunk-green" if chunk.is_relevant else "chunk-red"
                badge = "RELEVANT" if chunk.is_relevant else "IRRELEVANT"
                ground_truth = " · ground truth" if chunk.contains_ground_truth else ""
                st.markdown(
                    f"""
                    <div class="chunk-box {css_class}">
                        <strong>{_escape(chunk.chunk_id)}</strong> · {badge} · {chunk.relevance_score:.2f}{ground_truth}<br/>
                        {_escape(chunk.text)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown("**Claim audit**")
            if not result.claim_verifications:
                st.info("No answer claims to inspect.")
            for claim in result.claim_verifications:
                icon = "✓" if claim.status == "SUPPORTED" else ("✗" if claim.status == "HALLUCINATED" else "⚠")
                st.markdown(
                    f"{icon} **{_escape(claim.status.title())}** — {_escape(claim.claim)}  \n"
                    f"<small>{_escape(claim.reason)}</small>",
                    unsafe_allow_html=True,
                )
            if result.missing_ground_truth_facts:
                st.markdown("**Missing facts**")
                for fact in result.missing_ground_truth_facts:
                    st.markdown(f"- {_escape(fact)}")


def _question_options(test_cases: Iterable[SyntheticTestCase]) -> list[str]:
    return ["[Custom question]"] + [
        f"{case.id} · {case.question_type} · {case.question}" for case in test_cases
    ]


def render_overview(
    pipeline: DecoupledRAGPipeline,
    evaluator: RAGEvaluator,
    synthetic_tests: list[SyntheticTestCase],
    top_k: int,
    simulate_bad_retrieval: bool,
    simulate_mode: str,
    openai_api_key: Optional[str],
    run_evaluation: EvaluationRunner,
    threshold: float,
) -> None:
    render_animated_hero(
        "Overview / fast read",
        "Can we trust this answer?",
        "Ask one question. Read four scores. Open the evidence only when you need to explain the result.",
    )

    with st.container(border=True):
        st.markdown('<div class="section-label">1 · Ask a question</div>', unsafe_allow_html=True)
        options = _question_options(synthetic_tests)
        selected = st.selectbox(
            "Question library",
            options=options,
            index=1 if len(options) > 1 else 0,
            label_visibility="collapsed",
        )
        if selected == "[Custom question]":
            default_query = "How many books can a student borrow, and how long is the standard borrowing period?"
            default_ground_truth = "Students may borrow up to 5 books at one time, and the standard borrowing period is 14 days."
        else:
            selected_case = synthetic_tests[options.index(selected) - 1]
            default_query = selected_case.question
            default_ground_truth = selected_case.ground_truth
        query = st.text_input("Your question", value=default_query, key="overview_query")
        with st.expander("Edit reference answer (optional)"):
            ground_truth = st.text_area(
                "Reference answer",
                value=default_ground_truth,
                height=78,
                key="overview_ground_truth",
                help="Used for context recall and answer accuracy.",
            )

    result = run_evaluation(query, ground_truth.strip() or None)
    st.session_state.latest_query = query
    st.session_state.latest_ground_truth = ground_truth.strip() or None
    st.session_state.latest_evaluation = result

    st.markdown('<div class="section-label">2 · Read the result</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    metrics = [
        ("Relevance", result.context_relevance, "search"),
        ("Recall", result.context_recall, "search"),
        ("Faithfulness", result.faithfulness, "answer"),
        ("Accuracy", result.answer_accuracy, "answer"),
    ]
    for column, (title, score, subtitle) in zip(columns, metrics):
        with column:
            render_scorecard(title, score, subtitle, threshold)
    render_verdict(result)
    render_answer_card(result, "3 · Final answer")
    render_evidence(result, "See evidence · retrieved context and claim audit")


def _chat_status(result: EvaluationResult) -> str:
    report = result.root_cause
    if report is None:
        return "No diagnosis"
    if report.verdict_code == "PASS":
        return f"Grounded · {result.faithfulness:.0%} faithful"
    return f"Needs review · {report.verdict_code.replace('_', ' ').title()}"


def render_chat(
    synthetic_tests: list[SyntheticTestCase],
    run_evaluation: EvaluationRunner,
    threshold: float,
) -> None:
    render_page_header(
        "User view / simple chat",
        "Ask campus AI",
        "A clean assistant experience for students. The diagnostic details stay out of the way until you need them.",
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if not st.session_state.chat_history:
        st.session_state.chat_history.append(
            {
                "query": "How many books can a student borrow, and how long is the standard borrowing period?",
                "ground_truth": "Students may borrow up to 5 books at one time, and the standard borrowing period is 14 days.",
            }
        )

    st.markdown('<div class="chat-shell fade-in">', unsafe_allow_html=True)
    for item in st.session_state.chat_history:
        result = run_evaluation(item["query"], item.get("ground_truth"))
        with st.chat_message("user"):
            st.markdown(item["query"])
        with st.chat_message("assistant"):
            st.markdown(result.generated_answer)
            st.markdown(
                f'<div class="chat-meta"><span class="chat-pill">{_escape(_chat_status(result))}</span>'
                f'<span>{len(result.chunk_evaluations)} context chunks checked</span></div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    example_prompts = [
        "What is the library late fee?",
        "When are hostel quiet hours?",
        "How early should I arrive for an exam?",
    ]
    example_columns = st.columns(3)
    for index, (column, prompt) in enumerate(zip(example_columns, example_prompts)):
        with column:
            if st.button(prompt, key=f"chat_example_{index}", use_container_width=True):
                st.session_state.chat_history.append({"query": prompt, "ground_truth": None})
                st.rerun()

    prompt = st.chat_input("Ask about the library, hostel, exams, or shuttle...")
    if prompt:
        ground_truth = None
        lowered_prompt = prompt.lower()
        for case in synthetic_tests:
            if case.question.lower() == lowered_prompt:
                ground_truth = case.ground_truth
                break
        st.session_state.chat_history.append({"query": prompt, "ground_truth": ground_truth})
        st.rerun()


def _problem_list(result: EvaluationResult, threshold: float) -> list[str]:
    problems = []
    retriever_score = result.context_relevance
    if result.context_recall is not None:
        retriever_score = min(retriever_score, result.context_recall)
    if retriever_score < threshold:
        problems.append(f"Retriever score is {retriever_score:.0%}; the needed evidence may not have been retrieved.")
    if result.faithfulness < threshold:
        problems.append(f"Faithfulness is {result.faithfulness:.0%}; one or more answer claims are unsupported.")
    if result.answer_accuracy is not None and result.answer_accuracy < threshold:
        problems.append(f"Answer accuracy is {result.answer_accuracy:.0%}; the answer is incomplete or incorrect.")
    return problems or ["No active problems. This run is inside the configured quality threshold."]


def render_engineer(
    current_query: str,
    current_ground_truth: Optional[str],
    run_evaluation: EvaluationRunner,
    threshold: float,
) -> None:
    render_page_header(
        "Engineer view / diagnostics",
        "Find the failure fast",
        "This is the technical view: what the retriever found, what the generator said, and what to fix next.",
    )
    result = run_evaluation(current_query, current_ground_truth)
    st.markdown(
        f'<div class="active-run"><span>Active query</span><strong>{_escape(current_query)}</strong></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    metrics = [
        ("Relevance", result.context_relevance, "retriever"),
        ("Recall", result.context_recall, "retriever"),
        ("Faithfulness", result.faithfulness, "generator"),
        ("Accuracy", result.answer_accuracy, "end-to-end"),
    ]
    for column, (title, score, subtitle) in zip(columns, metrics):
        with column:
            render_scorecard(title, score, subtitle, threshold)
    render_verdict(result)

    st.markdown('<div class="section-label">Problems to fix</div>', unsafe_allow_html=True)
    problems = _problem_list(result, threshold)
    for problem in problems:
        st.markdown(f'<div class="problem-row">{_escape(problem)}</div>', unsafe_allow_html=True)

    if result.root_cause:
        st.markdown('<div class="section-label">Recommended remediation</div>', unsafe_allow_html=True)
        for step in result.root_cause.remediation_steps:
            st.markdown(f'<div class="remediation-row">{_escape(step)}</div>', unsafe_allow_html=True)
    render_answer_card(result, "Generated answer · claim attribution")
    render_evidence(result, "Open trace · context and sentence-level audit")


def render_batch_audit(
    pipeline: DecoupledRAGPipeline,
    evaluator: RAGEvaluator,
    synthetic_tests: list[SyntheticTestCase],
    top_k: int,
    simulate_bad_retrieval: bool,
    simulate_mode: str,
    openai_api_key: Optional[str],
) -> None:
    with st.expander("Run batch audit · optional"):
        st.caption("Evaluate every generated question with the current scenario.")
        if st.button("Run batch audit", use_container_width=True):
            rows = []
            for case in synthetic_tests:
                chunks = pipeline.retrieve(case.question, top_k=top_k, simulate_bad_retrieval=simulate_bad_retrieval)
                answer = pipeline.generate(
                    case.question,
                    chunks,
                    openai_api_key=openai_api_key or None,
                    simulate_mode=simulate_mode,
                )
                result = evaluator.evaluate(
                    case.question,
                    chunks,
                    answer,
                    case.ground_truth,
                    openai_api_key=openai_api_key or None,
                )
                rows.append(
                    {
                        "ID": case.id,
                        "Type": case.question_type,
                        "Relevance": f"{result.context_relevance:.0%}",
                        "Recall": f"{result.context_recall:.0%}" if result.context_recall is not None else "N/A",
                        "Faithfulness": f"{result.faithfulness:.0%}",
                        "Accuracy": f"{result.answer_accuracy:.0%}" if result.answer_accuracy is not None else "N/A",
                        "Diagnosis": result.root_cause.verdict_code if result.root_cause else "N/A",
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
