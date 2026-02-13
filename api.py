from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core import run_rag

app = FastAPI(title="RAG Studio API", version="0.1.0")


class RunRequest(BaseModel):
    query: str = Field(..., description="User query.")
    config: Dict[str, Any] | None = Field(default=None, description="Retrieval and model config.")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/runs")
def run_endpoint(payload: RunRequest) -> Dict[str, Any]:
    clean_query = payload.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        return run_rag(clean_query, payload.config or {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
