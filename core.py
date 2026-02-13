from __future__ import annotations

import copy
import hashlib
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal

RetrieverType = Literal["bm25", "dense", "hybrid"]

DEFAULT_CONFIG: Dict[str, Any] = {
    "project": "RAG Studio Demo",
    "dataset": "Default Dataset",
    "index": "main-index",
    "model": "gpt-4.1-mini",
    "retriever": "hybrid",
    "top_k": 6,
    "threshold": 0.2,
    "hybrid_alpha": 0.5,
    "reranker_enabled": False,
    "reranker_model": "cross-encoder-mini",
    "query_rewrite": False,
    "multi_query": False,
    "n_queries": 1,
    "mmr": False,
    "chunking": {
        "chunk_size": 512,
        "overlap": 64,
        "chunker_version": "v1",
    },
}

DUMMY_CORPUS: List[Dict[str, Any]] = [
    {
        "doc": "retrieval_playbook.pdf",
        "page": 3,
        "section": "BM25 Basics",
        "chunk_id": "retrieval_playbook:p3:c1",
        "snippet": (
            "BM25 is a sparse retriever that rewards lexical matches and term frequency. "
            "It often works well for keyword-heavy enterprise queries."
        ),
    },
    {
        "doc": "retrieval_playbook.pdf",
        "page": 7,
        "section": "Dense Retrieval",
        "chunk_id": "retrieval_playbook:p7:c2",
        "snippet": (
            "Dense retrieval maps queries and chunks into embeddings to capture semantic "
            "similarity beyond exact token overlap."
        ),
    },
    {
        "doc": "hybrid_search_notes.md",
        "page": 1,
        "section": "Hybrid Alpha",
        "chunk_id": "hybrid_search_notes:p1:c1",
        "snippet": (
            "Hybrid retrieval interpolates sparse and dense scores. "
            "The alpha parameter controls the weight between lexical and semantic signals."
        ),
    },
    {
        "doc": "reranking_guide.md",
        "page": 2,
        "section": "Re-ranking",
        "chunk_id": "reranking_guide:p2:c4",
        "snippet": (
            "A cross-encoder reranker can improve top-k precision by rescoring candidate chunks "
            "with stronger query-document interaction."
        ),
    },
    {
        "doc": "query_ops_handbook.txt",
        "page": 5,
        "section": "Query Rewrite",
        "chunk_id": "query_ops_handbook:p5:c3",
        "snippet": (
            "Query rewriting can expand abbreviations and normalize intent so retrieval systems "
            "match more relevant passages."
        ),
    },
    {
        "doc": "query_ops_handbook.txt",
        "page": 6,
        "section": "Multi-query",
        "chunk_id": "query_ops_handbook:p6:c2",
        "snippet": (
            "Multi-query retrieval samples multiple paraphrases and merges candidates, "
            "which can increase recall for ambiguous prompts."
        ),
    },
    {
        "doc": "chunking_reference.pdf",
        "page": 10,
        "section": "Chunk Size Tradeoffs",
        "chunk_id": "chunking_reference:p10:c5",
        "snippet": (
            "Larger chunks increase context continuity but may reduce precision. "
            "Overlap helps preserve references split across boundaries."
        ),
    },
    {
        "doc": "evaluation_manual.md",
        "page": 4,
        "section": "Citation Quality",
        "chunk_id": "evaluation_manual:p4:c1",
        "snippet": (
            "Grounded answers should cite the exact supporting chunk and avoid unsupported "
            "claims not present in retrieved evidence."
        ),
    },
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_int(value: Any, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _stable_unit_float(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _tokenize(text: str) -> List[str]:
    clean = "".join(char.lower() if char.isalnum() else " " for char in text)
    return [token for token in clean.split() if token]


def _lexical_overlap(query_tokens: List[str], text_tokens: List[str]) -> float:
    if not query_tokens or not text_tokens:
        return 0.0
    query_set = set(query_tokens)
    text_set = set(text_tokens)
    overlap = len(query_set.intersection(text_set))
    return overlap / max(1, len(query_set))


def _dense_similarity(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    t_tokens = _tokenize(text)
    overlap = _lexical_overlap(q_tokens, t_tokens)
    length_factor = 1.0 - abs(len(q_tokens) - len(t_tokens)) / max(1, len(q_tokens) + len(t_tokens))
    return max(0.0, min(1.0, 0.6 * overlap + 0.4 * length_factor))


def normalize_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = copy.deepcopy(DEFAULT_CONFIG)
    if not config:
        return normalized

    for key in ("project", "dataset", "index", "model", "reranker_model"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            normalized[key] = value.strip()

    retriever = str(config.get("retriever", normalized["retriever"])).lower()
    if retriever in {"bm25", "dense", "hybrid"}:
        normalized["retriever"] = retriever

    normalized["top_k"] = _clamp_int(config.get("top_k"), 3, 20, normalized["top_k"])
    normalized["threshold"] = _clamp_float(config.get("threshold"), 0.0, 1.0, normalized["threshold"])
    normalized["hybrid_alpha"] = _clamp_float(
        config.get("hybrid_alpha"), 0.0, 1.0, normalized["hybrid_alpha"]
    )
    normalized["reranker_enabled"] = bool(config.get("reranker_enabled", normalized["reranker_enabled"]))
    normalized["query_rewrite"] = bool(config.get("query_rewrite", normalized["query_rewrite"]))
    normalized["multi_query"] = bool(config.get("multi_query", normalized["multi_query"]))
    normalized["n_queries"] = _clamp_int(config.get("n_queries"), 1, 5, normalized["n_queries"])
    normalized["mmr"] = bool(config.get("mmr", normalized["mmr"]))

    chunking = config.get("chunking")
    if isinstance(chunking, dict):
        normalized["chunking"]["chunk_size"] = _clamp_int(
            chunking.get("chunk_size"), 64, 8192, normalized["chunking"]["chunk_size"]
        )
        normalized["chunking"]["overlap"] = _clamp_int(
            chunking.get("overlap"), 0, 2048, normalized["chunking"]["overlap"]
        )
        chunker_version = chunking.get("chunker_version")
        if isinstance(chunker_version, str) and chunker_version.strip():
            normalized["chunking"]["chunker_version"] = chunker_version.strip()

    if not normalized["multi_query"]:
        normalized["n_queries"] = 1

    return normalized


def _score_chunk(
    chunk: Dict[str, Any],
    scoring_queries: List[str],
    retriever: RetrieverType,
    alpha: float,
) -> Dict[str, float]:
    best_bm25 = 0.0
    best_dense = 0.0
    corpus_text = f"{chunk['section']} {chunk['snippet']}"

    for sub_query in scoring_queries:
        query_tokens = _tokenize(sub_query)
        text_tokens = _tokenize(corpus_text)
        lexical = _lexical_overlap(query_tokens, text_tokens)
        bm25 = 0.85 * lexical + 0.15 * _stable_unit_float(f"bm25:{sub_query}:{chunk['chunk_id']}")
        dense = (
            0.55 * lexical
            + 0.35 * _dense_similarity(sub_query, corpus_text)
            + 0.10 * _stable_unit_float(f"dense:{sub_query}:{chunk['chunk_id']}")
        )
        best_bm25 = max(best_bm25, bm25)
        best_dense = max(best_dense, dense)

    if retriever == "bm25":
        combined = best_bm25
    elif retriever == "dense":
        combined = best_dense
    else:
        combined = alpha * best_bm25 + (1.0 - alpha) * best_dense

    return {
        "bm25": round(max(0.0, min(1.0, best_bm25)), 4),
        "dense": round(max(0.0, min(1.0, best_dense)), 4),
        "combined": round(max(0.0, min(1.0, combined)), 4),
    }


def _apply_mmr(hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_docs: set[str] = set()
    remaining = sorted(hits, key=lambda hit: hit["score"], reverse=True)

    while remaining and len(selected) < top_k:
        best_index = 0
        best_value = float("-inf")
        for idx, hit in enumerate(remaining):
            diversity_penalty = 0.12 if hit["doc"] in seen_docs else 0.0
            mmr_score = hit["score"] - diversity_penalty
            if mmr_score > best_value:
                best_index = idx
                best_value = mmr_score
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        seen_docs.add(chosen["doc"])

    return selected


def run_rag(query: str, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    started = time.perf_counter()
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("query must not be empty")

    final_config = normalize_config(config)
    retriever = final_config["retriever"]
    top_k = final_config["top_k"]
    threshold = final_config["threshold"]
    alpha = final_config["hybrid_alpha"]

    effective_query = clean_query
    if final_config["query_rewrite"]:
        effective_query = f"{clean_query} rewritten with normalized retrieval intent"

    generated_queries = [effective_query]
    if final_config["multi_query"]:
        generated_queries = [
            f"{effective_query} :: aspect {index + 1}"
            for index in range(final_config["n_queries"])
        ]

    hits: List[Dict[str, Any]] = []
    for chunk in DUMMY_CORPUS:
        score_pack = _score_chunk(chunk, generated_queries, retriever, alpha)
        score = score_pack["combined"]
        if score < threshold:
            continue

        hit = {
            "rank": 0,
            "score": score,
            "method": retriever,
            "doc": chunk["doc"],
            "page": chunk["page"],
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"],
            "snippet": chunk["snippet"],
            "badges": [retriever],
            "score_components": {
                "bm25": score_pack["bm25"],
                "dense": score_pack["dense"],
            },
        }
        hits.append(hit)

    hits.sort(key=lambda item: item["score"], reverse=True)
    if final_config["mmr"]:
        hits = _apply_mmr(hits, top_k)

    if final_config["reranker_enabled"]:
        query_tokens = set(_tokenize(effective_query))
        for hit in hits:
            section_tokens = set(_tokenize(hit["section"]))
            overlap = len(query_tokens.intersection(section_tokens))
            rerank_boost = min(0.08, overlap * 0.02)
            hit["score"] = round(min(1.0, hit["score"] + rerank_boost), 4)
            hit["method"] = "rerank"
            if "rerank" not in hit["badges"]:
                hit["badges"].append("rerank")
        hits.sort(key=lambda item: item["score"], reverse=True)

    hits = hits[:top_k]
    for rank, hit in enumerate(hits, start=1):
        hit["rank"] = rank

    citations: List[Dict[str, Any]] = []
    for idx, hit in enumerate(hits[:3], start=1):
        citations.append(
            {
                "id": idx,
                "doc": hit["doc"],
                "page": hit["page"],
                "section": hit["section"],
                "snippet": hit["snippet"],
                "chunk_id": hit["chunk_id"],
                "badges": hit["badges"],
            }
        )

    if citations:
        statements: List[str] = []
        for citation in citations:
            snippet_head = citation["snippet"].split(".")[0].strip()
            statements.append(
                f"{snippet_head} [{citation['id']}]"
            )
        answer = " ".join(statements)
    else:
        answer = "I could not find grounded evidence for this query in the current index."

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    latency_ms = max(35, elapsed_ms + 12 * len(hits) + (20 if final_config["reranker_enabled"] else 0))
    tokens_in = max(6, len(_tokenize(clean_query)) * 5 + len(generated_queries) * 4)
    tokens_out = max(16, len(_tokenize(answer)) * 2)
    cost = round(tokens_in * 0.000002 + tokens_out * 0.000004, 6)

    return {
        "run_id": str(uuid.uuid4()),
        "created_at": _utc_now_iso(),
        "query": clean_query,
        "effective_query": effective_query,
        "answer": answer,
        "hits": hits,
        "citations": citations,
        "final_config": final_config,
        "metrics": {
            "latency_ms": latency_ms,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost": cost,
        },
        "debug": {
            "generated_queries": generated_queries,
            "retriever": retriever,
            "threshold": threshold,
        },
    }
