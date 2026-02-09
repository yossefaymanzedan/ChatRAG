from __future__ import annotations

import hashlib
import logging
import threading
import uuid
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.database import Database
from app.embeddings import EmbeddingService
from app.llm import DeepSeekClient
from app.parsers import SUPPORTED_EXTS, parse_file, preview_text, token_count
from app.vector_store import VectorStore


def _build_index_logger() -> logging.Logger:
    logger = logging.getLogger("chatrag.indexer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    log_path = Path(".rag") / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


INDEX_LOGGER = _build_index_logger()


@dataclass
class JobState:
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    files_total: int = 0
    files_processed: int = 0
    chunks_indexed: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    detail: str | None = None


class IndexJobManager:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embeddings: EmbeddingService,
        llm: DeepSeekClient | None = None,
    ) -> None:
        self.db = db
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._embedding_batch_size = max(1, int(settings.embedding_batch_size))
        self._upsert_retries = max(1, int(settings.index_vector_upsert_retries))
        self._upsert_retry_delay_ms = max(0, int(settings.index_vector_upsert_retry_delay_ms))

    def _log(self, job_id: str, message: str, level: int = logging.INFO) -> None:
        INDEX_LOGGER.log(level, f"job={job_id} | {message}")

    def start_index(self, folder_path: str) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = JobState(job_id=job_id, status="queued", detail="Queued")
        self._log(job_id, f"queued | folder_path={folder_path}")

        thread = threading.Thread(target=self._run_index, args=(job_id, folder_path), daemon=True)
        thread.start()
        return job_id

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for k, v in kwargs.items():
                setattr(job, k, v)

    def _run_index(self, job_id: str, folder_path: str) -> None:
        root = Path(folder_path).resolve()
        if not root.exists() or not root.is_dir():
            self._update(job_id, status="failed", detail="Folder does not exist", errors=[f"Invalid folder path: {folder_path}"])
            self._log(job_id, f"failed | invalid folder path={folder_path}", logging.ERROR)
            return

        self._update(job_id, status="running", detail="Scanning files")
        self._log(job_id, f"start | root={root}")

        try:
            total_chunks = 0
            files = [
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            ]
            files = sorted(files)
            self._update(job_id, files_total=len(files), detail="Indexing files")
            self._log(job_id, f"scan_complete | files_total={len(files)}")

            existing_docs = {
                row["file_path"]: row
                for row in self.db.get_documents_by_root(str(root))
            }
            seen_paths: set[str] = set()

            for i, file_path in enumerate(files, start=1):
                self._update(job_id, detail=f"Indexing {i}/{len(files)}: {file_path.name}")
                abs_path = str(file_path.resolve())
                seen_paths.add(abs_path)
                file_size = 0
                try:
                    file_size = file_path.stat().st_size
                except Exception:
                    file_size = 0
                file_started = time.perf_counter()
                self._log(
                    job_id,
                    f"file_start | idx={i}/{len(files)} | name={file_path.name} | size_bytes={file_size}",
                )
                file_hash = self._sha256(file_path)
                modified = datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                existing = existing_docs.get(abs_path)

                if (
                    existing
                    and existing["file_hash"] == file_hash
                    and self.db.document_has_anchor_type(existing["id"], "doc_summary")
                ):
                    total_chunks += len(self.db.get_chunk_ids_for_document(existing["id"]))
                    self._update(job_id, chunks_indexed=total_chunks)
                    self._update_progress(job_id, i)
                    elapsed = time.perf_counter() - file_started
                    self._log(
                        job_id,
                        f"file_skip_cached | idx={i}/{len(files)} | name={file_path.name} | elapsed_sec={elapsed:.2f}",
                    )
                    continue

                doc_id: str | None = None
                try:
                    doc_id = self.db.upsert_document(
                        file_path=abs_path,
                        file_name=file_path.name,
                        file_ext=file_path.suffix.lower().lstrip("."),
                        file_hash=file_hash,
                        modified_time=modified,
                        status="processing",
                    )

                    if existing:
                        old_chunk_ids = self.db.get_chunk_ids_for_document(doc_id)
                        self.vector_store.delete(old_chunk_ids)
                        self._log(
                            job_id,
                            f"file_cleanup_old_vectors | idx={i}/{len(files)} | name={file_path.name} | old_chunks={len(old_chunk_ids)}",
                        )

                    self._update(job_id, detail=f"Parsing {i}/{len(files)}: {file_path.name}")
                    parse_started = time.perf_counter()
                    parse = parse_file(
                        file_path,
                        pdf_text_threshold=settings.pdf_text_threshold,
                        ignore_front_matter=settings.ignore_front_matter,
                        front_matter_scan_pages=settings.front_matter_scan_pages,
                    )
                    if parse.warnings:
                        self._append_warnings(job_id, parse.warnings)
                    parse_elapsed = time.perf_counter() - parse_started
                    self._log(
                        job_id,
                        f"file_parsed | idx={i}/{len(files)} | name={file_path.name} | chunks={len(parse.chunks)} | warnings={len(parse.warnings)} | elapsed_sec={parse_elapsed:.2f}",
                    )

                    if not parse.chunks:
                        self.db.set_document_status(doc_id, "skipped", "No parseable chunks")
                        self._append_warnings(job_id, [f"{file_path.name}: no parseable chunks"])
                        self._update_progress(job_id, i)
                        elapsed = time.perf_counter() - file_started
                        self._log(
                            job_id,
                            f"file_skipped_no_chunks | idx={i}/{len(files)} | name={file_path.name} | elapsed_sec={elapsed:.2f}",
                            logging.WARNING,
                        )
                        continue

                    final_chunks = self._augment_with_document_summary(file_path, parse.chunks)
                    chunk_ids = self.db.replace_document_chunks(doc_id, final_chunks)
                    self._log(
                        job_id,
                        f"file_chunks_prepared | idx={i}/{len(files)} | name={file_path.name} | final_chunks={len(final_chunks)}",
                    )

                    for start in range(0, len(final_chunks), self._embedding_batch_size):
                        end = start + self._embedding_batch_size
                        chunk_slice = final_chunks[start:end]
                        id_slice = chunk_ids[start:end]
                        batch_no = (start // self._embedding_batch_size) + 1
                        batch_total = (len(final_chunks) + self._embedding_batch_size - 1) // self._embedding_batch_size
                        texts = [c["text"] for c in chunk_slice]
                        embed_started = time.perf_counter()
                        embeddings = self.embeddings.embed(texts)
                        embed_elapsed = time.perf_counter() - embed_started

                        metadatas = []
                        for chunk_id, chunk in zip(id_slice, chunk_slice, strict=False):
                            metadatas.append(
                                {
                                    "chunk_id": chunk_id,
                                    "document_id": doc_id,
                                    "file_path": abs_path,
                                    "anchor_type": chunk["anchor_type"],
                                    "anchor_page": chunk.get("anchor_page"),
                                    "anchor_section": chunk.get("anchor_section"),
                                    "anchor_paragraph": chunk.get("anchor_paragraph"),
                                    "anchor_row": chunk.get("anchor_row"),
                                }
                            )

                        upsert_started = time.perf_counter()
                        self._upsert_with_retry(
                            ids=id_slice,
                            embeddings=embeddings,
                            metadatas=metadatas,
                            documents=texts,
                        )
                        upsert_elapsed = time.perf_counter() - upsert_started
                        self._log(
                            job_id,
                            (
                                f"file_batch_done | idx={i}/{len(files)} | name={file_path.name} "
                                f"| batch={batch_no}/{batch_total} | batch_size={len(chunk_slice)} "
                                f"| embed_sec={embed_elapsed:.2f} | upsert_sec={upsert_elapsed:.2f}"
                            ),
                        )

                    self.db.save_vector_map((cid, cid) for cid in chunk_ids)
                    self.db.set_document_status(doc_id, "processed")
                    total_chunks += len(chunk_ids)
                    self._update(job_id, chunks_indexed=total_chunks)
                    elapsed = time.perf_counter() - file_started
                    self._log(
                        job_id,
                        (
                            f"file_done | idx={i}/{len(files)} | name={file_path.name} "
                            f"| chunks_indexed={len(chunk_ids)} | total_chunks={total_chunks} | elapsed_sec={elapsed:.2f}"
                        ),
                    )
                except Exception as exc:
                    if doc_id:
                        self.db.set_document_status(
                            doc_id,
                            "failed",
                            f"Indexing failed for {file_path.name}: {exc}",
                        )
                    self._append_errors(job_id, [f"{file_path.name}: {exc}"])
                    self._log(
                        job_id,
                        (
                            f"file_failed | idx={i}/{len(files)} | name={file_path.name} "
                            f"| error={exc} | traceback={traceback.format_exc()}"
                        ),
                        logging.ERROR,
                    )
                finally:
                    self._update_progress(job_id, i)

            deleted = [path for path, row in existing_docs.items() if path not in seen_paths]
            for path in deleted:
                row = existing_docs[path]
                old_chunk_ids = self.db.get_chunk_ids_for_document(row["id"])
                self.vector_store.delete(old_chunk_ids)
                self.db.delete_document(row["id"])
                self._log(job_id, f"file_deleted_missing | path={path} | old_chunks={len(old_chunk_ids)}")

            self._update(job_id, status="completed", progress=100.0, detail="Indexing completed")
            self._log(job_id, "completed | progress=100")
        except Exception as exc:
            self._append_errors(job_id, [str(exc)])
            self._update(job_id, status="failed", detail="Indexing failed")
            self._log(
                job_id,
                f"job_failed | error={exc} | traceback={traceback.format_exc()}",
                logging.ERROR,
            )

    def _augment_with_document_summary(self, file_path: Path, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return chunks
        summary_chunk = self._build_document_summary_chunk(file_path, chunks)
        if not summary_chunk:
            return chunks
        merged = [summary_chunk] + list(chunks)
        for idx, ch in enumerate(merged):
            ch["chunk_index"] = idx
        return merged

    def _build_document_summary_chunk(self, file_path: Path, chunks: list[dict]) -> dict | None:
        max_points = max(1, int(settings.index_summary_max_points))
        excerpt_chars = max(80, int(settings.index_summary_excerpt_chars))
        char_budget = max(400, int(settings.index_summary_char_budget))
        skip_llm_threshold = max(0, int(settings.index_summary_skip_llm_chunk_threshold))

        sample_parts: list[str] = []
        for ch in chunks[:max_points]:
            part = (ch.get("preview") or ch.get("text") or "").strip()
            if part:
                sample_parts.append(part[:excerpt_chars])
        if not sample_parts:
            return None
        trimmed_parts: list[str] = []
        used = 0
        for part in sample_parts:
            next_len = len(part) + 3
            if used + next_len > char_budget and trimmed_parts:
                break
            trimmed_parts.append(part)
            used += next_len
        sample_text = "\n".join(f"- {p}" for p in trimmed_parts)
        sample_words = sample_text.split()
        if len(sample_words) > 500:
            sample_text = " ".join(sample_words[:500])
            sample_words = sample_text.split()

        summary_text = ""
        skip_llm = len(chunks) > skip_llm_threshold
        INDEX_LOGGER.info(
            "summary_prepare | file=%s | chunks=%s | sample_words=%s | skip_llm=%s",
            file_path.name,
            len(chunks),
            len(sample_words),
            skip_llm,
        )
        try:
            if self.llm and self.llm.is_configured() and not skip_llm:
                system = (
                    "Summarize a document for retrieval indexing. "
                    "Return strict JSON with keys summary and topics. "
                    "summary: 2-4 sentences, concrete and factual. "
                    "topics: array of 4-10 short keywords."
                )
                user = (
                    f"File name: {file_path.name}\n"
                    "Representative excerpts:\n"
                    f"{sample_text}\n"
                )
                llm_started = time.perf_counter()
                data = self.llm.chat_json(system, user, timeout=float(settings.index_summary_timeout_sec))
                llm_elapsed = time.perf_counter() - llm_started
                summary = str(data.get("summary", "")).strip()
                topics = data.get("topics", [])
                if not isinstance(topics, list):
                    topics = []
                cleaned_topics = [str(t).strip() for t in topics if str(t).strip()]
                topic_line = f"Topics: {', '.join(cleaned_topics[:10])}" if cleaned_topics else ""
                summary_text = "\n".join(x for x in [summary, topic_line] if x).strip()
                INDEX_LOGGER.info(
                    "summary_llm_done | file=%s | elapsed_sec=%.2f | topics=%s",
                    file_path.name,
                    llm_elapsed,
                    len(cleaned_topics),
                )
        except Exception:
            summary_text = ""
            INDEX_LOGGER.warning("summary_llm_failed | file=%s", file_path.name, exc_info=True)

        if not summary_text:
            summary_text = f"Document summary for {file_path.name}:\n" + " ".join(sample_parts[:4])
            INDEX_LOGGER.info("summary_fallback_used | file=%s", file_path.name)

        return {
            "chunk_index": 0,
            "text": summary_text,
            "token_count": token_count(summary_text),
            "anchor_type": "doc_summary",
            "anchor_page": None,
            "anchor_section": "Document summary",
            "anchor_paragraph": 0,
            "anchor_row": None,
            "start_char": None,
            "end_char": None,
            "preview": preview_text(summary_text),
        }

    def _update_progress(self, job_id: str, current_idx: int) -> None:
        job = self.get(job_id)
        if not job:
            return
        total = max(job.files_total, 1)
        progress = min(100.0, (current_idx / total) * 100.0)
        self._update(job_id, files_processed=current_idx, progress=progress)

    def _append_warnings(self, job_id: str, items: list[str]) -> None:
        with self._lock:
            self._jobs[job_id].warnings.extend(items)

    def _append_errors(self, job_id: str, items: list[str]) -> None:
        with self._lock:
            self._jobs[job_id].errors.extend(items)

    def _upsert_with_retry(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
        documents: list[str],
    ) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, self._upsert_retries + 1):
            try:
                self.vector_store.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents,
                )
                return
            except Exception as exc:
                last_exc = exc
                INDEX_LOGGER.warning(
                    "vector_upsert_retry | attempt=%s/%s | error=%s",
                    attempt,
                    self._upsert_retries,
                    exc,
                )
                if attempt >= self._upsert_retries:
                    break
                if self._upsert_retry_delay_ms > 0:
                    time.sleep(self._upsert_retry_delay_ms / 1000.0)
        if last_exc:
            raise last_exc

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
