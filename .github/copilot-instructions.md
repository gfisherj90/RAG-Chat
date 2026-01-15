# Copilot / AI Agent Instructions

Purpose: Short, actionable guidance so an AI coding agent can be immediately productive in this RAG chatbot codebase.

- **Big picture**: FastAPI backend (`backend/app.py`) powers a small web UI (`frontend/`) and a RAG orchestrator (`backend/rag_system.py`). Core components:
  - `DocumentProcessor` (`backend/document_processor.py`) — parses `docs/*` files (expects header lines and `Lesson N:` markers) and produces text chunks.
  - `VectorStore` (`backend/vector_store.py`) — ChromaDB persistent collections: `course_catalog` (metadata) and `course_content` (chunks). Uses `sentence-transformers` for embeddings.
  - `AIGenerator` (`backend/ai_generator.py`) — wraps Anthropic Claude; system prompt enforces: one search per course-specific query, brief answers, and no meta commentary.
  - `ToolManager` / `CourseSearchTool` (`backend/search_tools.py`) — exposes a tool-based search interface; tools must implement `get_tool_definition()` with a `name` field.
  - `SessionManager` (`backend/session_manager.py`) — in-memory session history; `create_session()` returns `session_<n>` IDs.

- **Primary data flow**:
  1. Add docs to `docs/` (plain text with header lines). Example format:
     ```text
     Course Title: Intro to AI
     Course Link: https://example
     Course Instructor: Dr. Name

     Lesson 1: Overview
     Lesson Link: https://...
     <lesson text...>
     ```
  2. `DocumentProcessor.process_course_document()` -> splits into chunks using `CHUNK_SIZE`/`CHUNK_OVERLAP` from `backend/config.py`.
  3. `VectorStore.add_course_metadata()` and `add_course_content()` -> persist into ChromaDB (`CHROMA_PATH`).
  4. Query path: frontend -> POST `/api/query` -> `RAGSystem.query()` -> `AIGenerator.generate_response()` (may invoke tools) -> tool executes `VectorStore.search()` -> results returned and stored as `last_sources` on the search tool.

- **Dev / run workflows**:
  - Install: `uv sync` (project uses `uv`), set `.env` with `ANTHROPIC_API_KEY`.
  - Quick start: `chmod +x run.sh && ./run.sh` — which runs `uv run uvicorn app:app --reload --port 8000` from `backend/`.
  - Manual dev: `cd backend && uv run uvicorn app:app --reload --port 8000`.
  - API docs available at `http://localhost:8000/docs` when running.

- **Project-specific conventions & gotchas**:
  - Course title string is used as the Chroma collection ID / course ID (`ids=[course.title]`). Avoid renaming course titles without rebuilding the DB.
  - Metadata stores `lessons_json` as a JSON string in `course_catalog.metadatas` and is parsed back by `VectorStore.get_all_courses_metadata()`.
  - `DocumentProcessor` relies on `Lesson N:` markers and falls back to treating the remainder as a single document when no lessons are found.
  - `AIGenerator.SYSTEM_PROMPT` expects the model to return `tool_use` stop reasons for tool-based searches — tool execution and follow-up happens inside `AIGenerator._handle_tool_execution()`.
  - `SessionManager` keeps a very small in-memory history (`MAX_HISTORY` from `config.py`). For long running or multi-process deployments, replace with persistent storage.

- **Integration points & external deps**:
  - Anthropic Claude: `ANTHROPIC_API_KEY` required in `.env` (`backend/config.py`). Model string in config: `ANTHROPIC_MODEL`.
  - ChromaDB persistent client: `chroma_db` files written under `CHROMA_PATH` (default `./chroma_db`). Embedding via `sentence-transformers` (`EMBEDDING_MODEL`).
  - Frontend calls `/api/query` and `/api/courses`; static files served from `frontend/` via FastAPI `StaticFiles` in `backend/app.py`.

- **When editing behavior / adding features** (concrete examples):
  - To change chunking, edit `CHUNK_SIZE` and `CHUNK_OVERLAP` in `backend/config.py` and the `chunk_text()` logic in `backend/document_processor.py`.
  - To add a new tool, create a class implementing `Tool` in `backend/search_tools.py` with `get_tool_definition()` returning a `name`, and register it in `RAGSystem.__init__` with `tool_manager.register_tool(...)`.
  - To change how sources are surfaced to the UI, modify `CourseSearchTool._format_results()` to change `last_sources` content and `frontend/script.js` where sources are rendered.

- **Files to inspect first when debugging or extending**:
  - `backend/rag_system.py` — orchestration and public methods used by API.
  - `backend/document_processor.py` — parsing and chunking rules.
  - `backend/vector_store.py` — Chroma queries, filters (`course_title`, `lesson_number`) and metadata schema.
  - `backend/ai_generator.py` and `backend/search_tools.py` — AI toolflow and tool execution.
  - `backend/config.py` — all tunable constants (embedding model, chunk sizes, API keys).

If anything here is unclear or you want additional examples (e.g., sample course file, unit-test skeleton for `DocumentProcessor`, or a migration strategy for renaming course titles), tell me which section to expand and I will iterate.
