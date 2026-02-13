# RAG Studio (Streamlit MVP)

Minimal, clean Python project for a "RAG Studio" style UI:

- Streamlit WebUI with 2-pane layout
- Optional FastAPI `POST /runs` endpoint
- Direct core fallback (UI works without API)
- JSONL run and feedback logging

## Files

- `app.py` - Streamlit UI
- `core.py` - `run_rag(query, config) -> dict` (dummy retrieval, final schema)
- `api.py` - FastAPI endpoint delegating to `core.run_rag`
- `storage.py` - JSONL append/read helpers
- `README.md` - setup and usage

## Requirements

- Python 3.10+

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install streamlit fastapi uvicorn pydantic requests
```

## Start WebUI (direct core mode, default)

```bash
streamlit run app.py
```

This starts the app in direct mode (`core.run_rag`) with no API needed.

## Optional API mode

Start API server:

```bash
uvicorn api:app --reload
```

Run Streamlit with API enabled:

```bash
USE_API=true API_URL=http://127.0.0.1:8000/runs streamlit run app.py
```

If API fails, the UI falls back to direct core calls automatically.

## Usage

1. Select `Project`, `Dataset`, `Index`, `Model` in top bar.
2. Configure retrieval in `Retrieval` tab.
3. Enter query in `User Input`, click `Run` or `Send / Run`.
4. Inspect citations in `Sources`.
5. Inspect hit table and `effective_query` in `Debug`.
6. Submit `Hilfreich` / `Nicht hilfreich` feedback in `Eval/Logs`.
7. Open `Logs Drawer` for latency/tokens/cost, copy run JSON, and download `storage.jsonl`.

## Data Logging

Events are appended to `storage.jsonl` as JSON Lines:

- `event = "run"` on each run
- `event = "feedback"` on each feedback action (linked by `run_id`)

Stored fields include `final_config`, `hits`, `citations`, `answer`, `latency_ms`, and `feedback`.
# BA
