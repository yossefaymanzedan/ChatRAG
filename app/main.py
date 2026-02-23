from pathlib import Path
import sys
import json
from uuid import uuid4
import re
import shutil
import threading
import webbrowser
import os
import socket

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Allow running as `python app/main.py` in addition to module mode.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.chat_service import ChatService
from app.config import ensure_runtime_dirs, settings
from app.database import Database
from app.embeddings import EmbeddingService
from app.indexer import IndexJobManager
from app.llm import DeepSeekClient
from app.models import (
    ChatRequest,
    ChatResponse,
    ChunkResponse,
    ClearIndexResponse,
    CreateChatRequest,
    RenameChatRequest,
    ChatMessageResponse,
    ChatSessionResponse,
    IndexRequest,
    IndexResponse,
    IndexStatusResponse, 
    SaveMessageRequest,
    DeleteUploadFileRequest,
    DeleteUploadFileResponse,
    UploadStatusResponse,
    UploadFilesResponse,
    UploadFileItem,
)
from app.retrieval import HybridRetriever
from app.vector_store import VectorStore


ensure_runtime_dirs()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}

app = FastAPI(title="ChatRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(settings.sqlite_path)
db.init_schema()
vector_store = VectorStore(settings.chroma_dir, settings.chroma_collection)
embeddings = EmbeddingService(
    settings.embedding_model,
    cache_dir=settings.embedding_cache_dir,
    offline=settings.embedding_offline,
)
retriever = HybridRetriever(db, vector_store)
llm = DeepSeekClient(
    provider=settings.llm_provider,
    base_url=settings.ollama_base_url,
    model=settings.ollama_model,
    temperature=settings.ollama_temperature,
    openai_api_key=settings.openai_api_key,
    openai_model=settings.openai_model,
    openai_base_url=settings.openai_base_url,
    deepseek_api_key=settings.deepseek_api_key,
    deepseek_model=settings.deepseek_model,
    deepseek_base_url=settings.deepseek_base_url,
)
index_jobs = IndexJobManager(db, vector_store, embeddings, llm)
chat_service = ChatService(db, embeddings, retriever, llm)
app.mount("/static", StaticFiles(directory="static"), name="static")

AUTO_OPEN_URL = f"http://{os.getenv('CHATRAG_HOST', '127.0.0.1')}:{_env_int('CHATRAG_PORT', 8080)}"


@app.on_event("startup")
def open_browser_after_startup() -> None:
    if getattr(app.state, "browser_open_scheduled", False):
        return
    app.state.browser_open_scheduled = True
    threading.Timer(0.2, lambda: webbrowser.open(AUTO_OPEN_URL)).start()


def _upload_has_indexed_content(upload_id: str | None) -> bool:
    if not upload_id:
        return False
    docs = db.get_documents_by_upload_id(upload_id)
    if not docs:
        return False
    for doc in docs:
        if db.get_chunk_ids_for_document(doc["id"]):
            return True
    return False


def _require_upload_with_content(upload_id: str | None) -> None:
    if not upload_id:
        raise HTTPException(status_code=400, detail="No upload selected. Please upload files first.")
    if not _upload_has_indexed_content(upload_id):
        raise HTTPException(status_code=400, detail="No indexed files found for this upload. Please upload/index files.")


def _upload_counts(upload_id: str | None) -> tuple[int, int]:
    if not upload_id:
        return 0, 0
    docs = db.get_documents_by_upload_id(upload_id)
    chunk_total = 0
    for doc in docs:
        chunk_total += len(db.get_chunk_ids_for_document(doc["id"]))
    return len(docs), chunk_total


def _unique_upload_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path
    stem = target_path.stem
    suffix = target_path.suffix
    parent = target_path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


async def _write_upload_file(upload: UploadFile, target_path: Path, chunk_size: int = 1024 * 1024) -> None:
    with target_path.open("wb") as out:
        while True:
            chunk = await upload.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/settings/llm")
def get_llm_settings():
    return llm.get_runtime_config()


@app.post("/settings/llm")
def update_llm_settings(payload: dict = Body(...)):
    try:
        return llm.update_runtime_config(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/settings/ollama/models")
def list_ollama_models():
    try:
        return {
            "models": llm.ollama_list_models(),
            "recommended": [
                {"name": "llama3.2:1b", "label": "Llama 3.2 1B"},
                {"name": "qwen2.5:3b", "label": "Qwen 2.5 3B"},
                {"name": "qwen2.5:7b", "label": "Qwen 2.5 7B"},
                {"name": "qwen2.5:14b", "label": "Qwen 2.5 14B"},
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to read Ollama models: {exc}") from exc


@app.post("/settings/ollama/pull")
def pull_ollama_model(payload: dict = Body(...)):
    model_name = str(payload.get("model") or "").strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="model is required")

    def generate():
        try:
            for item in llm.ollama_pull_model(model_name):
                yield f"event: progress\n"
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            try:
                current = llm.get_runtime_config()
                current["ollama"] = current.get("ollama") or {}
                current["ollama"]["model"] = model_name
                llm.update_runtime_config({"ollama": {"model": model_name}})
            except Exception:
                pass
            yield "event: done\n"
            yield f"data: {json.dumps({'ok': True, 'model': model_name}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            yield "event: error\n"
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/")
def root():
    index_path = Path("static/index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "ChatRAG API is running", "docs": "/docs", "health": "/health"}


@app.post("/index", response_model=IndexResponse)
def start_index(request: IndexRequest) -> IndexResponse:
    job_id = index_jobs.start_index(request.folder_path)
    return IndexResponse(job_id=job_id)


@app.post("/index/upload", response_model=IndexResponse)
async def upload_and_index_folder(
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
    upload_id: str | None = Form(default=None),
    start_indexing: bool = Form(default=True),
) -> IndexResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    safe_upload_id = upload_id or str(uuid4())
    safe_upload_id = re.sub(r"[^a-zA-Z0-9\\-_]", "", safe_upload_id) or str(uuid4())
    upload_root = Path(".rag") / "uploads" / safe_upload_id
    upload_root.mkdir(parents=True, exist_ok=True)

    for idx, file in enumerate(files):
        rel = paths[idx] if idx < len(paths) and paths[idx] else (file.filename or f"file_{idx}")
        rel = rel.replace("\\", "/").lstrip("/")
        rel_path = Path(rel)
        safe_parts = [p for p in rel_path.parts if p not in ("..", ".", "")]
        final_path = upload_root.joinpath(*safe_parts)
        final_path = _unique_upload_path(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        await _write_upload_file(file, final_path)
        await file.close()

    if not start_indexing:
        return IndexResponse(job_id=None, upload_id=safe_upload_id)

    job_id = index_jobs.start_index(str(upload_root.resolve()))
    return IndexResponse(job_id=job_id, upload_id=safe_upload_id)


@app.post("/index/upload/start", response_model=IndexResponse)
def start_index_for_upload(payload: dict = Body(...)) -> IndexResponse:
    upload_id = str(payload.get("upload_id") or "").strip()
    if not upload_id:
        raise HTTPException(status_code=400, detail="upload_id is required")
    safe_upload_id = re.sub(r"[^a-zA-Z0-9\\-_]", "", upload_id)
    if not safe_upload_id:
        raise HTTPException(status_code=400, detail="invalid upload_id")
    upload_root = (Path(".rag") / "uploads" / safe_upload_id).resolve()
    if not upload_root.exists() or not upload_root.is_dir():
        raise HTTPException(status_code=404, detail="upload folder not found")
    job_id = index_jobs.start_index(str(upload_root))
    return IndexResponse(job_id=job_id, upload_id=safe_upload_id)


@app.get("/index/status", response_model=IndexStatusResponse)
def get_index_status(job_id: str = Query(...)) -> IndexStatusResponse:
    job = index_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    effective_status = job.status
    effective_progress = job.progress
    effective_detail = job.detail
    # Safety normalization: if all files are processed but state did not flip yet,
    # treat it as completed so the UI cannot be stuck in "running".
    if (
        job.status == "running"
        and job.files_total > 0
        and job.files_processed >= job.files_total
    ):
        effective_status = "completed"
        effective_progress = 100.0
        effective_detail = "Indexing completed"
    return IndexStatusResponse(
        job_id=job.job_id,
        status=effective_status,
        progress=effective_progress,
        files_total=job.files_total,
        files_processed=job.files_processed,
        chunks_indexed=job.chunks_indexed,
        warnings=job.warnings,
        errors=job.errors,
        detail=effective_detail,
        ocr=job.ocr,
    )


@app.get("/uploads/{upload_id}/status", response_model=UploadStatusResponse)
def get_upload_status(upload_id: str) -> UploadStatusResponse:
    doc_count, chunk_count = _upload_counts(upload_id)
    ocr_stats = db.get_upload_ocr_stats(upload_id)
    return UploadStatusResponse(
        upload_id=upload_id,
        has_indexed_content=chunk_count > 0,
        document_count=doc_count,
        chunk_count=chunk_count,
        ocr=ocr_stats,
    )


@app.get("/uploads/{upload_id}/files", response_model=UploadFilesResponse)
def list_upload_files(upload_id: str) -> UploadFilesResponse:
    docs = db.get_documents_by_upload_id(upload_id)
    items: list[UploadFileItem] = []
    for doc in docs:
        doc_id = str(doc["id"])
        file_path = str(doc["file_path"])
        chunk_count = len(db.get_chunk_ids_for_document(doc_id))
        file_size_bytes: int | None = None
        try:
            p = Path(file_path)
            if p.exists() and p.is_file():
                file_size_bytes = p.stat().st_size
        except Exception:
            file_size_bytes = None
        raw_status = str(doc["status"] or "queued")
        if chunk_count > 0:
            display_status = "processed"
        elif raw_status in {"indexed", "processing", "running", "indexing"}:
            display_status = "processing"
        else:
            display_status = raw_status
        items.append(
            UploadFileItem(
                name=str(doc["file_name"]),
                status=display_status,
                type=str(doc["file_ext"] or "file"),
                chunk_count=chunk_count,
                file_size_bytes=file_size_bytes,
                file_path=file_path,
            )
        )
    items.sort(key=lambda x: x.name.lower())
    return UploadFilesResponse(upload_id=upload_id, files=items)


@app.get("/uploads/{upload_id}/files/open")
def open_upload_file(upload_id: str, file_name: str = Query(...)):
    target_name = (file_name or "").strip()
    if not target_name:
        raise HTTPException(status_code=400, detail="file_name is required")
    docs = db.get_documents_by_upload_id(upload_id)
    matches = [doc for doc in docs if str(doc["file_name"]) == target_name]
    if not matches:
        raise HTTPException(status_code=404, detail="file not found in upload scope")
    # If duplicated names exist in nested folders, open the first existing file path.
    for doc in matches:
        p = Path(str(doc["file_path"]))
        if p.exists() and p.is_file():
            return FileResponse(path=p)
    raise HTTPException(status_code=404, detail="source file not found")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not llm.is_configured():
        raise HTTPException(status_code=503, detail="LLM is not configured. Open Settings and configure provider/model.")
    if request.mode != "general":
        _require_upload_with_content(request.upload_id)
    try:
        result = chat_service.chat(request.message, request.mode, request.upload_id, request.chat_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(**result)


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    if not llm.is_configured():
        raise HTTPException(status_code=503, detail="LLM is not configured. Open Settings and configure provider/model.")
    if request.mode != "general":
        _require_upload_with_content(request.upload_id)

    def generate():
        try:
            for item in chat_service.chat_stream(request.message, request.mode, request.upload_id, request.chat_id):
                yield f"event: {item['event']}\n"
                yield f"data: {json.dumps(item['data'], ensure_ascii=False)}\n\n"
        except RuntimeError as exc:
            payload = {"error": str(exc)}
            yield "event: error\n"
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/chunks/{chunk_id}", response_model=ChunkResponse)
def get_chunk(chunk_id: str) -> ChunkResponse:
    row = db.get_chunk(chunk_id)
    if not row:
        raise HTTPException(status_code=404, detail="chunk_id not found")
    item = dict(row)
    try:
        p = Path(item.get("file_path", ""))
        item["file_size_bytes"] = p.stat().st_size if p.exists() else None
    except Exception:
        item["file_size_bytes"] = None
    return ChunkResponse(**item)


@app.get("/chunks/{chunk_id}/file")
def open_chunk_file(chunk_id: str):
    row = db.get_chunk(chunk_id)
    if not row:
        raise HTTPException(status_code=404, detail="chunk_id not found")
    item = dict(row)
    p = Path(item.get("file_path", ""))
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="source file not found")
    return FileResponse(path=p)


@app.post("/clear_index", response_model=ClearIndexResponse)
def clear_index() -> ClearIndexResponse:
    db.clear_all()
    vector_store.reset_collection()
    return ClearIndexResponse(ok=True)


@app.post("/clear_all", response_model=ClearIndexResponse)
def clear_all() -> ClearIndexResponse:
    db.clear_everything()
    vector_store.reset_collection()
    uploads_dir = Path(".rag") / "uploads"
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir, ignore_errors=True)
    return ClearIndexResponse(ok=True)


@app.post("/chats", response_model=ChatSessionResponse)
def create_chat(req: CreateChatRequest) -> ChatSessionResponse:
    title = (req.title or "New Chat").strip() or "New Chat"
    row = db.create_chat_session(title, req.upload_id)
    return ChatSessionResponse(**dict(row))


@app.get("/chats", response_model=list[ChatSessionResponse])
def list_chats() -> list[ChatSessionResponse]:
    return [ChatSessionResponse(**dict(r)) for r in db.list_chat_sessions()]


@app.get("/chats/{chat_id}/messages", response_model=list[ChatMessageResponse])
def list_chat_messages(chat_id: str) -> list[ChatMessageResponse]:
    if not db.get_chat_session(chat_id):
        raise HTTPException(status_code=404, detail="chat_id not found")
    rows = db.list_chat_messages(chat_id)
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["citations"] = json.loads(item.get("citations_json") or "[]")
        except Exception:
            item["citations"] = []
        out.append(ChatMessageResponse(
            id=item["id"],
            chat_id=item["chat_id"],
            role=item["role"],
            content=item["content"],
            citations=item["citations"],
            created_at=item["created_at"],
        ))
    return out


@app.patch("/chats/{chat_id}", response_model=ChatSessionResponse)
def rename_chat(chat_id: str, req: RenameChatRequest) -> ChatSessionResponse:
    session = db.get_chat_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat_id not found")
    title = (req.title or "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    db.update_chat_title(chat_id, title)
    updated = db.get_chat_session(chat_id)
    return ChatSessionResponse(**dict(updated))


@app.post("/chats/{chat_id}/rename", response_model=ChatSessionResponse)
def rename_chat_legacy(chat_id: str, req: RenameChatRequest) -> ChatSessionResponse:
    return rename_chat(chat_id, req)


@app.post("/chats/{chat_id}/messages", response_model=ChatMessageResponse)
def save_chat_message(chat_id: str, req: SaveMessageRequest) -> ChatMessageResponse:
    session = db.get_chat_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat_id not found")
    row = db.add_chat_message(chat_id, req.role, req.content, json.dumps(req.citations, ensure_ascii=False))
    item = dict(row)
    return ChatMessageResponse(
        id=item["id"],
        chat_id=item["chat_id"],
        role=item["role"],
        content=item["content"],
        citations=req.citations,
        created_at=item["created_at"],
    )


@app.delete("/chats/{chat_id}/messages/{message_id}/from_here")
def truncate_chat_from_message(chat_id: str, message_id: str):
    session = db.get_chat_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat_id not found")
    if not db.get_chat_message(chat_id, message_id):
        raise HTTPException(status_code=404, detail="message_id not found")
    deleted_count = db.delete_chat_messages_from(chat_id, message_id)
    return {"ok": True, "deleted_messages": deleted_count}


@app.post("/chats/{chat_id}/auto_title", response_model=ChatSessionResponse)
def auto_title_chat(chat_id: str) -> ChatSessionResponse:
    session = db.get_chat_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat_id not found")
    msgs = db.list_chat_messages(chat_id)
    if not msgs:
        return ChatSessionResponse(**dict(session))

    convo = []
    for m in msgs[:6]:
        role = m["role"]
        content = m["content"]
        convo.append(f"{role.upper()}: {content}")
    prompt = "\n".join(convo)

    title = "New Chat"
    if llm.is_configured():
        try:
            data = llm.chat_json(
                "Generate a concise chat title from the conversation. Return JSON with key title only.",
                f"Conversation:\n{prompt}\nMax 6 words.",
                timeout=30.0,
            )
            title = str(data.get("title", "")).strip() or title
        except Exception:
            pass

    if title == "New Chat":
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "New Chat")
        title = " ".join(first_user.split()[:6]).strip() or "New Chat"

    db.update_chat_title(chat_id, title)
    updated = db.get_chat_session(chat_id)
    return ChatSessionResponse(**dict(updated))


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    session = db.get_chat_session(chat_id)
    if not session:
        raise HTTPException(status_code=404, detail="chat_id not found")
    upload_id = session["upload_id"] if "upload_id" in session.keys() else None

    # Purge indexed data scoped to this chat's upload.
    if upload_id:
        docs = db.get_documents_by_upload_id(upload_id)
        for doc in docs:
            chunk_ids = db.get_chunk_ids_for_document(doc["id"])
            vector_store.delete(chunk_ids)
            db.delete_document(doc["id"])

        upload_dir = Path(".rag") / "uploads" / str(upload_id)
        if upload_dir.exists():
            shutil.rmtree(upload_dir, ignore_errors=True)

    db.delete_chat_session(chat_id)
    return {"ok": True}


@app.delete("/uploads/{upload_id}/files", response_model=DeleteUploadFileResponse)
def delete_upload_file(upload_id: str, req: DeleteUploadFileRequest) -> DeleteUploadFileResponse:
    safe_upload_id = (upload_id or "").strip()
    target_name = (req.file_name or "").strip()
    if not safe_upload_id:
        raise HTTPException(status_code=400, detail="upload_id is required")
    if not target_name:
        raise HTTPException(status_code=400, detail="file_name is required")

    docs = db.get_documents_by_upload_id(safe_upload_id)
    to_delete = []
    for doc in docs:
        file_name = str(doc["file_name"] if "file_name" in doc.keys() else "")
        if file_name == target_name:
            to_delete.append(doc)

    if not to_delete:
        raise HTTPException(status_code=404, detail="file not found in upload scope")

    removed = 0
    for doc in to_delete:
        chunk_ids = db.get_chunk_ids_for_document(doc["id"])
        if chunk_ids:
            vector_store.delete(chunk_ids)
        file_path = Path(str(doc["file_path"]))
        db.delete_document(doc["id"])
        removed += 1
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
        except Exception:
            pass

    return DeleteUploadFileResponse(ok=True, removed_documents=removed)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("CHATRAG_HOST", "127.0.0.1")
    start_port = _env_int("CHATRAG_PORT", 8080)
    port_tries = max(1, _env_int("CHATRAG_PORT_TRIES", 20))
    reload_enabled = _env_bool("CHATRAG_RELOAD", False)

    chosen_port: int | None = None
    for offset in range(port_tries):
        candidate = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
                chosen_port = candidate
                break
            except OSError:
                continue

    if chosen_port is None:
        print(
            f"Unable to bind any port in range {start_port}-{start_port + port_tries - 1} "
            f"on host {host}."
        )
        raise SystemExit(1)

    AUTO_OPEN_URL = f"http://{host}:{chosen_port}"
    print(f"Starting server at {AUTO_OPEN_URL}/docs")
    uvicorn.run("app.main:app", host=host, port=chosen_port, reload=reload_enabled)
