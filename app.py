from __future__ import annotations

import base64
import html
import json
import os
import re
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from core import DEFAULT_CONFIG, run_rag
from storage import DEFAULT_LOG_PATH, append_event, read_events

try:
    import requests
except Exception:  # pragma: no cover - optional dependency path
    requests = None


PROJECT_OPTIONS = ["RAG Studio Demo", "Support KB", "Policy KB"]
DATASET_OPTIONS = ["Default Dataset", "v1_documents", "v2_documents"]
INDEX_OPTIONS = ["main-index", "bm25-index", "dense-index"]
MODEL_OPTIONS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini"]
RERANKER_OPTIONS = ["cross-encoder-mini", "cross-encoder-base", "cohere-rerank-v3"]
NOT_HELPFUL_REASONS = [
    "missing_citation",
    "wrong_chunk",
    "no_answer",
    "hallucination",
    "other",
]
DEMO_ASSISTANT_ANSWER = "Derzeit ist keiner bei VW da.."
VW_LOGO_PATH = Path("image/Volkswagen_logo_2019.svg.png")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_logo_data_uri(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None

    mime_type, _ = guess_type(path.name)
    if not mime_type:
        mime_type = "image/png"
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _inject_vw_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --vw-navy: #001b4f;
            --vw-blue: #0046ad;
            --vw-electric: #0a72ff;
            --vw-ice: #eef4fb;
            --vw-line: #d5dfed;
            --vw-text: #000000;
            --vw-white: #ffffff;
        }

        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 8% 8%, #ffffff 0%, #f3f8ff 35%, #e8f0fb 100%);
        }

        [data-testid="stHeader"] {
            background: transparent;
            height: 0rem;
        }

        [data-testid="stHeader"] > div {
            height: 0rem;
            min-height: 0rem;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        .block-container {
            max-width: 1450px;
            padding-top: 0.35rem;
            padding-bottom: 1.2rem;
        }

        .vw-hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f3f8ff 100%);
            border-radius: 16px;
            padding: 0.85rem 1rem;
            color: var(--vw-text);
            border: 1px solid #bfd2ec;
            box-shadow: 0 8px 20px rgba(13, 31, 59, 0.08);
            margin-bottom: 0.8rem;
            margin-top: 0.45rem;
        }

        .vw-hero-left {
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }

        .vw-logo-wrap {
            width: 66px;
            height: 66px;
            border-radius: 999px;
            background: #ffffff;
            border: 1px solid #bfd2ec;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: inset 0 0 0 1px #ffffff;
            flex: 0 0 auto;
        }

        .vw-logo-img {
            width: 56px;
            height: 56px;
            object-fit: contain;
            display: block;
        }

        .vw-hero-copy {
            margin-top: 0.35rem;
        }

        .vw-hero-title {
            font-weight: 700;
            font-size: 1.2rem;
            letter-spacing: 0.01rem;
            color: #000000;
        }

        .vw-hero-sub {
            opacity: 0.92;
            font-size: 0.85rem;
            color: #000000;
        }

        .vw-chip {
            background: #e9f2ff;
            border: 1px solid #bfd2ec;
            border-radius: 999px;
            padding: 0.35rem 0.7rem;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
            color: #000000;
        }

        .vw-card {
            background: var(--vw-white);
            border: 1px solid var(--vw-line);
            border-radius: 12px;
            padding: 0.85rem 0.9rem;
            box-shadow: 0 3px 10px rgba(13, 31, 59, 0.05);
            margin-bottom: 0.4rem;
        }

        .vw-card-title {
            color: #000000;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .vw-card-meta {
            color: #000000;
            font-size: 0.85rem;
            margin-bottom: 0.35rem;
        }

        .vw-card-snippet {
            color: var(--vw-text);
            font-size: 0.9rem;
            line-height: 1.35;
            margin-bottom: 0.35rem;
        }

        .vw-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
        }

        .vw-badge {
            background: #e9f2ff;
            color: #000000;
            border: 1px solid #c4d8f7;
            border-radius: 999px;
            padding: 0.18rem 0.5rem;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.01rem;
        }

        .chat-shell {
            background:
                linear-gradient(180deg, #f7fbff 0%, #edf4ff 100%);
            border: 1px solid #c7d9ef;
            border-radius: 14px;
            padding: 0.75rem 0.7rem;
            max-height: 430px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 0.45rem;
        }

        .chat-shell::-webkit-scrollbar {
            width: 8px;
        }

        .chat-shell::-webkit-scrollbar-thumb {
            background: #c2d6ef;
            border-radius: 999px;
        }

        .chat-row {
            display: flex;
            width: 100%;
        }

        .chat-row-user {
            justify-content: flex-end;
        }

        .chat-row-assistant {
            justify-content: flex-start;
        }

        .chat-bubble {
            max-width: 82%;
            border-radius: 12px;
            border: 1px solid #c7d9ef;
            box-shadow: 0 2px 8px rgba(13, 31, 59, 0.06);
            padding: 0.46rem 0.62rem;
        }

        .chat-bubble-user {
            background: #e8f1ff;
            border-bottom-right-radius: 5px;
        }

        .chat-bubble-assistant {
            background: #ffffff;
            border-bottom-left-radius: 5px;
        }

        .chat-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.72rem;
            color: #334155;
            margin-bottom: 0.18rem;
            gap: 0.5rem;
        }

        .chat-text {
            color: #000000;
            line-height: 1.4;
            font-size: 0.92rem;
            word-break: break-word;
        }

        .chat-empty {
            color: #334155;
            text-align: center;
            padding: 0.9rem 0.4rem;
            font-size: 0.9rem;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #c7d9ef;
            border-radius: 14px;
            padding: 0.5rem 0.55rem;
            margin-top: 0.55rem;
        }

        div[data-testid="stForm"] [data-testid="stTextInput"] input {
            border: none !important;
            border-radius: 999px;
            background: #fbfdff;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            padding-left: 0.85rem;
            height: 2.5rem;
            outline: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stForm"] [data-baseweb="input"] {
            box-shadow: none !important;
            outline: none !important;
            border: none !important;
        }

        div[data-testid="stForm"] [data-baseweb="input"] > div {
            border: 1px solid #c7d9ef !important;
            box-shadow: none !important;
            outline: none !important;
            background: #fbfdff !important;
            border-radius: 999px !important;
        }

        div[data-testid="stForm"] [data-baseweb="input"] > div:focus-within {
            border-color: #7ea7e4 !important;
            box-shadow: 0 0 0 1px rgba(126, 167, 228, 0.25) !important;
            outline: none !important;
        }

        div[data-testid="stForm"] [data-testid="stTextInput"] input:focus,
        div[data-testid="stForm"] [data-testid="stTextInput"] input:focus-visible {
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }

        div[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {
            color: #64748b !important;
            opacity: 1;
        }

        .stFormSubmitButton > button {
            height: 2.5rem;
            border-radius: 999px;
            font-weight: 700;
            color: #ffffff !important;
            background: linear-gradient(180deg, #001b4f 0%, #001338 100%) !important;
            border: 1px solid #001338 !important;
        }

        .stFormSubmitButton > button:hover {
            color: #ffffff !important;
            background: linear-gradient(180deg, #00266d 0%, #001b4f 100%) !important;
            border-color: #001b4f !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px;
        }

        div[data-testid="stMetric"] {
            background: var(--vw-white);
            border: 1px solid var(--vw-line);
            border-radius: 12px;
            padding: 0.4rem 0.5rem;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span,
        [data-testid="stAppViewContainer"] li,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] h4 {
            color: #000000;
        }

        div[data-testid="stForm"] .stFormSubmitButton > button,
        div[data-testid="stForm"] .stFormSubmitButton > button span,
        div[data-testid="stForm"] .stFormSubmitButton > button p {
            color: #ffffff !important;
        }

        div[data-testid="stTabs"] button[role="tab"] {
            border-bottom: 2px solid transparent;
            color: #000000;
            border-radius: 0;
            margin-right: 0.45rem;
        }

        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: #000000;
            border-bottom-color: var(--vw-electric);
            font-weight: 700;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #7ea7e4;
            background: linear-gradient(180deg, #ffffff 0%, #f2f7ff 100%);
            color: #000000;
            font-weight: 600;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #0a72ff;
            color: #000000;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        [data-testid="stTextArea"] textarea {
            border-radius: 10px;
            border-color: #bdd1eb;
            background-color: #fbfdff;
            color: #000000 !important;
            -webkit-text-fill-color: #000000 !important;
            caret-color: #000000 !important;
        }

        [data-testid="stTextArea"] textarea::placeholder {
            color: #556070 !important;
            opacity: 1;
        }

        /* Force black text in select controls and dropdown option lists */
        div[data-baseweb="select"] *,
        div[data-baseweb="input"] * {
            color: #000000 !important;
        }

        div[role="listbox"] *,
        ul[role="listbox"] *,
        [data-baseweb="menu"] * {
            color: #000000 !important;
        }

        @media (max-width: 900px) {
            .vw-hero {
                flex-direction: column;
                align-items: flex-start;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    logo_data_uri = _load_logo_data_uri(VW_LOGO_PATH)
    if logo_data_uri:
        logo_html = f"<img class='vw-logo-img' src='{logo_data_uri}' alt='VW Logo' />"
    else:
        logo_html = "<span style='font-weight:700;color:#000000;'>VW</span>"

    st.markdown(
        f"""
        <div class="vw-hero">
            <div class="vw-hero-left">
                <div class="vw-logo-wrap">
                    {logo_html}
                </div>
                <div class="vw-hero-copy">
                    <div class="vw-hero-title">VW RAG Studio</div>
                    <div class="vw-hero-sub">Retrieval Tuning, Citations and Evaluation Workspace</div>
                </div>
            </div>
            <div class="vw-chip">Design Language: Volkswagen Blue</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    st.session_state.setdefault("chat_messages", [])
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("selected_citation_id", None)
    st.session_state.setdefault(
        "feedback_state",
        {"label": None, "reason": "missing_citation", "comment": ""},
    )
    st.session_state.setdefault("show_hits_json", False)
    st.session_state.setdefault("run_notice", "")

    st.session_state.setdefault(
        "system_prompt",
        "You are a grounded assistant. Answer with citations based on retrieved evidence only.",
    )
    st.session_state.setdefault("compose_query", "")

    st.session_state.setdefault("project", DEFAULT_CONFIG["project"])
    st.session_state.setdefault("dataset", DEFAULT_CONFIG["dataset"])
    st.session_state.setdefault("index_name", DEFAULT_CONFIG["index"])
    st.session_state.setdefault("model_name", DEFAULT_CONFIG["model"])

    st.session_state.setdefault("retriever", DEFAULT_CONFIG["retriever"])
    st.session_state.setdefault("top_k", DEFAULT_CONFIG["top_k"])
    st.session_state.setdefault("threshold", float(DEFAULT_CONFIG["threshold"]))
    st.session_state.setdefault("hybrid_alpha", float(DEFAULT_CONFIG["hybrid_alpha"]))
    st.session_state.setdefault("reranker_enabled", DEFAULT_CONFIG["reranker_enabled"])
    st.session_state.setdefault("reranker_model", DEFAULT_CONFIG["reranker_model"])
    st.session_state.setdefault("query_rewrite", DEFAULT_CONFIG["query_rewrite"])
    st.session_state.setdefault("multi_query", DEFAULT_CONFIG["multi_query"])
    st.session_state.setdefault("n_queries", DEFAULT_CONFIG["n_queries"])
    st.session_state.setdefault("mmr", DEFAULT_CONFIG["mmr"])


def _collect_config(project: str, dataset: str, index_name: str, model_name: str) -> Dict[str, Any]:
    return {
        "project": project,
        "dataset": dataset,
        "index": index_name,
        "model": model_name,
        "retriever": st.session_state.retriever,
        "top_k": int(st.session_state.top_k),
        "threshold": float(st.session_state.threshold),
        "hybrid_alpha": float(st.session_state.hybrid_alpha),
        "reranker_enabled": bool(st.session_state.reranker_enabled),
        "reranker_model": st.session_state.reranker_model,
        "query_rewrite": bool(st.session_state.query_rewrite),
        "multi_query": bool(st.session_state.multi_query),
        "n_queries": int(st.session_state.n_queries),
        "mmr": bool(st.session_state.mmr),
        "chunking": {
            "chunk_size": DEFAULT_CONFIG["chunking"]["chunk_size"],
            "overlap": DEFAULT_CONFIG["chunking"]["overlap"],
            "chunker_version": DEFAULT_CONFIG["chunking"]["chunker_version"],
        },
    }


def _build_run_event(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "event": "run",
        "run_id": result.get("run_id"),
        "created_at": result.get("created_at"),
        "query": result.get("query"),
        "effective_query": result.get("effective_query"),
        "final_config": result.get("final_config"),
        "hits": result.get("hits", []),
        "citations": result.get("citations", []),
        "answer": result.get("answer"),
        "latency_ms": result.get("metrics", {}).get("latency_ms"),
        "feedback": None,
    }


def _build_feedback_event(
    result: Dict[str, Any],
    label: str,
    reason: str | None,
    comment: str,
) -> Dict[str, Any]:
    feedback_payload: Dict[str, Any] = {"label": label}
    if reason:
        feedback_payload["reason"] = reason
    if comment.strip():
        feedback_payload["comment"] = comment.strip()

    return {
        "event": "feedback",
        "run_id": result.get("run_id"),
        "created_at": _utc_now_iso(),
        "query": result.get("query"),
        "effective_query": result.get("effective_query"),
        "final_config": result.get("final_config"),
        "hits": result.get("hits", []),
        "citations": result.get("citations", []),
        "answer": result.get("answer"),
        "latency_ms": result.get("metrics", {}).get("latency_ms"),
        "feedback": feedback_payload,
    }


def _run_backend(query: str, config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    use_api = os.getenv("USE_API", "false").strip().lower() in {"1", "true", "yes", "on"}
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000/runs")

    if use_api:
        if requests is None:
            st.session_state.run_notice = "USE_API=true but requests is unavailable. Falling back to core."
            return run_rag(query, config), "core-fallback"
        try:
            response = requests.post(
                api_url,
                json={"query": query, "config": config},
                timeout=20,
            )
            response.raise_for_status()
            st.session_state.run_notice = f"Run executed via API: {api_url}"
            return response.json(), "api"
        except Exception as exc:
            st.session_state.run_notice = f"API failed ({exc}); fallback to direct core call."
            return run_rag(query, config), "core-fallback"

    st.session_state.run_notice = "Run executed via direct core call."
    return run_rag(query, config), "core"


def _trim_snippet(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


def _chat_time_label() -> str:
    return datetime.now().strftime("%H:%M")


def _normalize_chat_content(raw_content: Any) -> str:
    text = str(raw_content)
    # Backward-compat: previous buggy renders stored chat HTML in message content.
    if "<div" in text and "chat-row" in text:
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        for prefix in ("You ", "Assistant "):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        text = re.sub(r"^\d{1,2}:\d{2}\s+", "", text)
    return text


def _render_chat_panel(messages: List[Dict[str, Any]]) -> None:
    if not messages:
        st.markdown(
            "<div class='chat-shell'><div class='chat-empty'>Start a conversation by sending a query.</div></div>",
            unsafe_allow_html=True,
        )
        return

    rows: List[str] = []
    for message in messages:
        role = "user" if message.get("role") == "user" else "assistant"
        row_class = "chat-row-user" if role == "user" else "chat-row-assistant"
        bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-assistant"
        label = "You" if role == "user" else "Assistant"

        message_text = _normalize_chat_content(message.get("content", ""))
        text_html = html.escape(message_text).replace("\n", "<br/>")
        time_html = html.escape(str(message.get("ts", "")))
        rows.append(
            (
                f"<div class='chat-row {row_class}'>"
                f"<div class='chat-bubble {bubble_class}'>"
                f"<div class='chat-meta'><span>{label}</span><span>{time_html}</span></div>"
                f"<div class='chat-text'>{text_html}</div>"
                "</div>"
                "</div>"
            )
        )

    st.markdown(f"<div class='chat-shell'>{''.join(rows)}</div>", unsafe_allow_html=True)


def _render_sources_tab(result: Dict[str, Any] | None) -> None:
    if not result:
        st.info("No run yet. Execute a query to inspect citations.")
        return

    citations = result.get("citations", [])
    if not citations:
        st.warning("No citations returned for this run.")
        return

    for citation in citations:
        badges = citation.get("badges", [])
        badges_html = "".join(
            f"<span class='vw-badge'>{html.escape(str(badge).upper())}</span>" for badge in badges
        )
        with st.container():
            st.markdown(
                f"""
                <div class="vw-card">
                    <div class="vw-card-title">[{citation['id']}] {html.escape(str(citation['doc']))}</div>
                    <div class="vw-card-meta">Page {citation['page']} | Section: {html.escape(str(citation['section']))}</div>
                    <div class="vw-card-snippet">{html.escape(_trim_snippet(citation.get("snippet", ""), limit=260))}</div>
                    <div class="vw-badges">{badges_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Open evidence", key=f"open_evidence_{citation['id']}"):
                st.session_state.selected_citation_id = citation["id"]
        st.divider()

    selected_id = st.session_state.selected_citation_id
    selected = next((item for item in citations if item["id"] == selected_id), None)
    with st.expander("Evidence Viewer", expanded=selected is not None):
        if not selected:
            st.caption("Select a citation card to open evidence details.")
            return
        st.markdown(
            f"**Document:** {selected['doc']}  \n"
            f"**Page:** {selected['page']}  \n"
            f"**Section:** {selected['section']}  \n"
            f"**Chunk ID:** `{selected['chunk_id']}`"
        )
        st.code(selected.get("snippet", ""), language="text")
        st.info("PDF viewer placeholder: render source page preview here in a full implementation.")


def _render_retrieval_tab() -> None:
    st.selectbox("Retriever", options=["bm25", "dense", "hybrid"], key="retriever")
    st.slider("Top-k", min_value=3, max_value=20, key="top_k")
    st.slider("Threshold", min_value=0.0, max_value=1.0, step=0.01, key="threshold")

    if st.session_state.retriever == "hybrid":
        st.slider("Hybrid alpha", min_value=0.0, max_value=1.0, step=0.01, key="hybrid_alpha")
    else:
        st.slider(
            "Hybrid alpha",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="hybrid_alpha",
            disabled=True,
        )

    st.toggle("Re-Ranker", key="reranker_enabled")
    if st.session_state.reranker_enabled:
        st.selectbox("reranker_model", options=RERANKER_OPTIONS, key="reranker_model")

    st.toggle("Query Rewrite", key="query_rewrite")
    st.toggle("Multi-Query", key="multi_query")
    if st.session_state.multi_query:
        st.slider("n_queries", min_value=1, max_value=5, key="n_queries")
    else:
        st.slider("n_queries", min_value=1, max_value=5, key="n_queries", disabled=True)

    st.toggle("MMR", key="mmr")

    st.markdown("### Chunking (read-only)")
    c1, c2, c3 = st.columns(3)
    c1.text_input("chunk_size", value=str(DEFAULT_CONFIG["chunking"]["chunk_size"]), disabled=True)
    c2.text_input("overlap", value=str(DEFAULT_CONFIG["chunking"]["overlap"]), disabled=True)
    c3.text_input(
        "chunker_version",
        value=str(DEFAULT_CONFIG["chunking"]["chunker_version"]),
        disabled=True,
    )
    st.caption("Reindex required if chunking values are changed (not implemented in this MVP).")


def _render_debug_tab(result: Dict[str, Any] | None) -> None:
    if not result:
        st.info("No debug data yet.")
        return

    st.markdown(f"**effective_query**: `{result.get('effective_query', '')}`")
    rows: List[Dict[str, Any]] = []
    for hit in result.get("hits", []):
        rows.append(
            {
                "rank": hit.get("rank"),
                "score": hit.get("score"),
                "method": hit.get("method"),
                "doc": hit.get("doc"),
                "page": hit.get("page"),
                "chunk_id": hit.get("chunk_id"),
                "snippet": _trim_snippet(hit.get("snippet", ""), limit=120),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if st.button("Copy hits JSON", key="copy_hits_json_button"):
        st.session_state.show_hits_json = True
    if st.session_state.show_hits_json:
        st.code(json.dumps(result.get("hits", []), indent=2), language="json")


def _render_eval_logs_tab(result: Dict[str, Any] | None) -> None:
    st.markdown("### Feedback")
    if result:
        c1, c2 = st.columns(2)
        if c1.button("Hilfreich", use_container_width=True):
            append_event(_build_feedback_event(result, label="helpful", reason=None, comment=""))
            st.session_state.feedback_state = {"label": "helpful", "reason": "missing_citation", "comment": ""}
            st.success("Feedback gespeichert.")

        if c2.button("Nicht hilfreich", use_container_width=True):
            st.session_state.feedback_state["label"] = "not_helpful"

        if st.session_state.feedback_state.get("label") == "not_helpful":
            reason = st.selectbox("reason", options=NOT_HELPFUL_REASONS, key="feedback_reason")
            comment = st.text_area("comment", key="feedback_comment")
            if st.button("Nicht hilfreich speichern"):
                append_event(
                    _build_feedback_event(
                        result,
                        label="not_helpful",
                        reason=reason,
                        comment=comment,
                    )
                )
                st.success("Feedback gespeichert.")
                st.session_state.feedback_state = {
                    "label": "submitted_not_helpful",
                    "reason": reason,
                    "comment": comment,
                }
    else:
        st.caption("Run a query first to submit evaluation feedback.")

    st.markdown("### Recent log events")
    events = read_events(limit=20)
    if not events:
        st.caption("No events logged yet.")
        return

    table_rows: List[Dict[str, Any]] = []
    for event in reversed(events):
        feedback = event.get("feedback") or {}
        table_rows.append(
            {
                "event": event.get("event"),
                "run_id": event.get("run_id"),
                "created_at": event.get("created_at"),
                "latency_ms": event.get("latency_ms"),
                "feedback_label": feedback.get("label"),
                "reason": feedback.get("reason"),
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def _render_bottom_drawer(result: Dict[str, Any] | None) -> None:
    with st.expander("Logs Drawer", expanded=False):
        if result:
            metrics = result.get("metrics", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("Latency (ms)", metrics.get("latency_ms"))
            c2.metric("Tokens", f"{metrics.get('tokens_in', 0)} in / {metrics.get('tokens_out', 0)} out")
            c3.metric("Cost", f"${metrics.get('cost', 0.0)}")

            st.markdown("**Copy JSON**")
            st.code(json.dumps(result, indent=2), language="json")
        else:
            st.caption("No run yet.")

        log_path = Path(DEFAULT_LOG_PATH)
        if log_path.exists():
            st.download_button(
                "Download run log",
                data=log_path.read_bytes(),
                file_name=log_path.name,
                mime="application/jsonl",
            )
        else:
            st.caption("No log file available yet.")


def main() -> None:
    st.set_page_config(page_title="RAG Studio", page_icon=":material/hub:", layout="wide")
    _init_state()
    _inject_vw_theme()
    _render_brand_header()

    st.markdown("### RAG Studio Console")
    if st.session_state.run_notice:
        st.caption(st.session_state.run_notice)

    top_left, top_mid, top_mid2, top_right, top_run, top_save = st.columns([2, 2, 2, 2, 1, 1])
    project = top_left.selectbox("Project", options=PROJECT_OPTIONS, key="project")
    dataset = top_mid.selectbox("Dataset", options=DATASET_OPTIONS, key="dataset")
    index_name = top_mid2.selectbox("Index", options=INDEX_OPTIONS, key="index_name")
    model_name = top_right.selectbox("Model", options=MODEL_OPTIONS, key="model_name")
    run_from_top = top_run.button("Run", use_container_width=True)
    save_payload = json.dumps(_collect_config(project, dataset, index_name, model_name), indent=2)
    top_save.download_button(
        "Save",
        data=save_payload,
        file_name="rag-config-preset.json",
        mime="application/json",
        use_container_width=True,
    )

    left_pane, right_pane = st.columns([3, 2], gap="large")
    send_from_chat = False
    compose_query_value = st.session_state.get("compose_query", "")

    with left_pane:
        with st.expander("System Prompt", expanded=False):
            st.text_area("system_prompt", key="system_prompt", height=120)

        st.markdown("### Chat")
        _render_chat_panel(st.session_state.chat_messages)

        with st.form("chat_compose_form", clear_on_submit=True, enter_to_submit=False):
            compose_col, send_col = st.columns([6, 1])
            with compose_col:
                compose_query_value = st.text_input(
                    "Message",
                    key="compose_query",
                    label_visibility="collapsed",
                    placeholder="Write a message...",
                )
            with send_col:
                send_from_chat = st.form_submit_button("Send", use_container_width=True)
        st.caption("Press Enter or click Send.")

    with right_pane:
        tab_sources, tab_retrieval, tab_debug, tab_eval = st.tabs(
            ["Sources", "Retrieval", "Debug", "Eval/Logs"]
        )
        with tab_sources:
            _render_sources_tab(st.session_state.last_result)
        with tab_retrieval:
            _render_retrieval_tab()
        with tab_debug:
            _render_debug_tab(st.session_state.last_result)
        with tab_eval:
            _render_eval_logs_tab(st.session_state.last_result)

    _render_bottom_drawer(st.session_state.last_result)

    if run_from_top or send_from_chat:
        query = compose_query_value.strip()
        if not query:
            st.warning("Please provide a query before running.")
            return

        run_config = _collect_config(project, dataset, index_name, model_name)
        result, _mode = _run_backend(query, run_config)
        result["answer"] = DEMO_ASSISTANT_ANSWER

        st.session_state.last_result = result
        st.session_state.show_hits_json = False
        st.session_state.feedback_state = {"label": None, "reason": "missing_citation", "comment": ""}
        st.session_state.selected_citation_id = result["citations"][0]["id"] if result.get("citations") else None
        st.session_state.chat_messages.append({"role": "user", "content": query, "ts": _chat_time_label()})
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": result.get("answer", ""), "ts": _chat_time_label()}
        )

        append_event(_build_run_event(result))
        st.rerun()


if __name__ == "__main__":
    main()
