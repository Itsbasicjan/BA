from __future__ import annotations

import json
import os
from datetime import datetime, timezone
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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    st.session_state.setdefault("user_query", "")

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


def _render_sources_tab(result: Dict[str, Any] | None) -> None:
    if not result:
        st.info("No run yet. Execute a query to inspect citations.")
        return

    citations = result.get("citations", [])
    if not citations:
        st.warning("No citations returned for this run.")
        return

    for citation in citations:
        badge_line = " ".join(f"`{badge}`" for badge in citation.get("badges", []))
        with st.container():
            st.markdown(
                f"**[{citation['id']}] {citation['doc']}**  \n"
                f"Page {citation['page']} | Section: {citation['section']}"
            )
            st.caption(_trim_snippet(citation.get("snippet", "")))
            if badge_line:
                st.markdown(badge_line)
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

    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RAG Studio")
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

    with left_pane:
        with st.expander("System Prompt", expanded=False):
            st.text_area("system_prompt", key="system_prompt", height=120)

        st.markdown("### Chat")
        if not st.session_state.chat_messages:
            st.caption("No messages yet.")
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.text_area("User Input", key="user_query", placeholder="Ask your retrieval question...", height=120)
        run_from_left = st.button("Send / Run")

        if st.session_state.last_result:
            st.markdown("### Latest answer")
            st.write(st.session_state.last_result.get("answer", ""))

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

    if run_from_top or run_from_left:
        query = st.session_state.user_query.strip()
        if not query:
            st.warning("Please provide a query before running.")
            return

        run_config = _collect_config(project, dataset, index_name, model_name)
        result, _mode = _run_backend(query, run_config)

        st.session_state.last_result = result
        st.session_state.show_hits_json = False
        st.session_state.feedback_state = {"label": None, "reason": "missing_citation", "comment": ""}
        st.session_state.selected_citation_id = result["citations"][0]["id"] if result.get("citations") else None
        st.session_state.chat_messages.append({"role": "user", "content": query})
        st.session_state.chat_messages.append({"role": "assistant", "content": result.get("answer", "")})

        append_event(_build_run_event(result))
        st.rerun()


if __name__ == "__main__":
    main()
