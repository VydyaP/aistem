# AI STEM Tutor

A modular AI tutor for STEM education, integrating Agentic AI, RAG, and advanced math reasoning. Built for the IBM Hackathon.

## Features
- Step-by-step math problem solving
- RAG (Retrieval-Augmented Generation) for grounded explanations
- Agentic orchestration for dynamic task routing
- Modular backend (FastAPI)

## Setup
From the **project root** (parent of `ai_stem_tutor/`). On macOS with Homebrew Python, use a virtual environment:

1. Create and use the venv:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r ai_stem_tutor/requirements.txt streamlit
   ```

## Structure
- `routers/`: API endpoints
- `services/`: Core logic (math, RAG, LLM)
- `models/`: Pydantic schemas
- `utils/`: Config and helpers


## How to run
From project root. Activate the venv first: `source .venv/bin/activate`.

1. **Backend:** `uvicorn ai_stem_tutor.main:app --reload --port 8001`
2. **Frontend:** `streamlit run streamlit_app.py`
3. **Ollama (LLM):** Start Ollama, then run `ollama list` and `ollama run granite3.3:8b`