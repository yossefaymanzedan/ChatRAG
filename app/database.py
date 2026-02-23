import sqlite3
import uuid
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


class Database:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path

    @contextmanager
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    modified_time TEXT NOT NULL,
                    indexed_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    anchor_type TEXT NOT NULL,
                    anchor_page INTEGER,
                    anchor_section TEXT,
                    anchor_paragraph INTEGER,
                    anchor_row INTEGER,
                    start_char INTEGER,
                    end_char INTEGER,
                    preview TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
                    chunk_id UNINDEXED,
                    text
                );

                CREATE TABLE IF NOT EXISTS vector_index_map (
                    chunk_id TEXT PRIMARY KEY,
                    vector_id TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    upload_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_anchor_page ON chunks(anchor_page);
                CREATE INDEX IF NOT EXISTS idx_chunks_anchor_paragraph ON chunks(anchor_paragraph);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat_id ON chat_messages(chat_id);
                """
            )
            self._ensure_chat_session_upload_id_column(conn)

    def clear_all(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                DELETE FROM fts_chunks;
                DELETE FROM vector_index_map;
                DELETE FROM chunks;
                DELETE FROM documents;
                """
            )

    def clear_everything(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                DELETE FROM fts_chunks;
                DELETE FROM vector_index_map;
                DELETE FROM chunks;
                DELETE FROM documents;
                DELETE FROM chat_messages;
                DELETE FROM chat_sessions;
                """
            )

    def get_documents_by_root(self, folder_path: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE file_path LIKE ?",
                (f"{folder_path}%",),
            )
            return list(cursor.fetchall())

    def get_document_by_path(self, file_path: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE file_path = ?", (file_path,))
            return cursor.fetchone()

    def upsert_document(
        self,
        *,
        file_path: str,
        file_name: str,
        file_ext: str,
        file_hash: str,
        modified_time: str,
        status: str,
        error_message: str | None = None,
    ) -> str:
        doc_id = str(uuid.uuid4())
        indexed_at = datetime.utcnow().isoformat()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM documents WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            if existing:
                doc_id = existing["id"]
                conn.execute(
                    """
                    UPDATE documents
                    SET file_name=?, file_ext=?, file_hash=?, modified_time=?, indexed_at=?, status=?, error_message=?
                    WHERE file_path=?
                    """,
                    (file_name, file_ext, file_hash, modified_time, indexed_at, status, error_message, file_path),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO documents(id, file_path, file_name, file_ext, file_hash, modified_time, indexed_at, status, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        file_path,
                        file_name,
                        file_ext,
                        file_hash,
                        modified_time,
                        indexed_at,
                        status,
                        error_message,
                    ),
                )
        return doc_id

    def set_document_status(self, doc_id: str, status: str, error_message: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE documents SET status = ?, error_message = ?, indexed_at = ? WHERE id = ?",
                (status, error_message, datetime.utcnow().isoformat(), doc_id),
            )

    def get_chunk_ids_for_document(self, document_id: str) -> list[str]:
        with self.connect() as conn:
            cursor = conn.execute("SELECT id FROM chunks WHERE document_id=?", (document_id,))
            return [row["id"] for row in cursor.fetchall()]

    def document_has_anchor_type(self, document_id: str, anchor_type: str) -> bool:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM chunks
                WHERE document_id = ? AND anchor_type = ?
                LIMIT 1
                """,
                (document_id, anchor_type),
            ).fetchone()
            return row is not None

    def delete_document(self, document_id: str) -> None:
        with self.connect() as conn:
            chunk_ids = [
                row["id"]
                for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,)).fetchall()
            ]
            if chunk_ids:
                placeholders = ",".join("?" for _ in chunk_ids)
                conn.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})", chunk_ids)
                conn.execute(f"DELETE FROM vector_index_map WHERE chunk_id IN ({placeholders})", chunk_ids)
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def replace_document_chunks(self, document_id: str, chunks: list[dict[str, Any]]) -> list[str]:
        created_at = datetime.utcnow().isoformat()
        with self.connect() as conn:
            old_ids = [
                row["id"]
                for row in conn.execute("SELECT id FROM chunks WHERE document_id = ?", (document_id,)).fetchall()
            ]
            if old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                conn.execute(f"DELETE FROM fts_chunks WHERE chunk_id IN ({placeholders})", old_ids)
                conn.execute(f"DELETE FROM vector_index_map WHERE chunk_id IN ({placeholders})", old_ids)
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

            inserted_ids: list[str] = []
            for item in chunks:
                chunk_id = str(uuid.uuid4())
                inserted_ids.append(chunk_id)
                conn.execute(
                    """
                    INSERT INTO chunks(
                        id, document_id, chunk_index, text, token_count,
                        anchor_type, anchor_page, anchor_section, anchor_paragraph,
                        anchor_row, start_char, end_char, preview, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_id,
                        document_id,
                        item["chunk_index"],
                        item["text"],
                        item["token_count"],
                        item["anchor_type"],
                        item.get("anchor_page"),
                        item.get("anchor_section"),
                        item.get("anchor_paragraph"),
                        item.get("anchor_row"),
                        item.get("start_char"),
                        item.get("end_char"),
                        item["preview"],
                        created_at,
                    ),
                )
                conn.execute(
                    "INSERT INTO fts_chunks(chunk_id, text) VALUES (?, ?)",
                    (chunk_id, item["text"]),
                )
            return inserted_ids

    def save_vector_map(self, pairs: Iterable[tuple[str, str]]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO vector_index_map(chunk_id, vector_id) VALUES(?, ?)",
                list(pairs),
            )

    def lexical_search(self, query: str, limit: int) -> list[sqlite3.Row]:
        fts_query = self._fts_escape(query)
        if not fts_query:
            return []
        with self.connect() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT chunk_id, bm25(fts_chunks) AS bm25_score
                    FROM fts_chunks
                    WHERE fts_chunks MATCH ?
                    ORDER BY bm25_score
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                return list(cursor.fetchall())
            except sqlite3.OperationalError:
                return []

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, d.file_path, d.file_name, d.file_ext
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()
        row_map = {row["id"]: row for row in rows}
        return [row_map[cid] for cid in chunk_ids if cid in row_map]

    def get_chunk(self, chunk_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT c.*, d.file_path, d.file_name, d.file_ext
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.id = ?
                """,
                (chunk_id,),
            ).fetchone()

    def get_diverse_chunks_for_documents(
        self,
        document_ids: list[str],
        per_doc: int = 3,
        limit_total: int = 10,
    ) -> list[sqlite3.Row]:
        if not document_ids:
            return []
        placeholders = ",".join("?" for _ in document_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, d.file_path, d.file_name, d.file_ext
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id IN ({placeholders})
                ORDER BY c.document_id, c.anchor_page, c.anchor_paragraph, c.chunk_index
                """,
                document_ids,
            ).fetchall()

        out: list[sqlite3.Row] = []
        doc_counts: dict[str, int] = {}
        seen_loc: set[str] = set()
        for row in rows:
            did = row["document_id"]
            if doc_counts.get(did, 0) >= per_doc:
                continue
            loc = f"{did}|{row['anchor_page']}|{row['anchor_paragraph']}|{row['anchor_row']}"
            if loc in seen_loc:
                continue
            seen_loc.add(loc)
            doc_counts[did] = doc_counts.get(did, 0) + 1
            out.append(row)
            if len(out) >= limit_total:
                break
        return out

    def get_chunks_for_document(self, document_id: str, limit: int = 300) -> list[sqlite3.Row]:
        if not document_id:
            return []
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT c.*, d.file_path, d.file_name, d.file_ext
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = ?
                ORDER BY c.anchor_page, c.anchor_paragraph, c.chunk_index
                LIMIT ?
                """,
                (document_id, int(limit)),
            ).fetchall()

    def create_chat_session(self, title: str, upload_id: str | None = None) -> sqlite3.Row:
        chat_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions(id, title, upload_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (chat_id, title, upload_id, now, now),
            )
            return conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (chat_id,)).fetchone()

    def list_chat_sessions(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM chat_sessions
                WHERE EXISTS (
                    SELECT 1
                    FROM chat_messages m
                    WHERE m.chat_id = chat_sessions.id
                )
                ORDER BY updated_at DESC
                """
            ).fetchall()

    def get_chat_session(self, chat_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (chat_id,)).fetchone()

    def update_chat_title(self, chat_id: str, title: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, datetime.utcnow().isoformat(), chat_id),
            )

    def add_chat_message(self, chat_id: str, role: str, content: str, citations_json: str) -> sqlite3.Row:
        msg_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages(id, chat_id, role, content, citations_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (msg_id, chat_id, role, content, citations_json, now),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (now, chat_id),
            )
            return conn.execute("SELECT * FROM chat_messages WHERE id = ?", (msg_id,)).fetchone()

    def list_chat_messages(self, chat_id: str) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM chat_messages
                WHERE chat_id = ?
                ORDER BY created_at ASC
                """,
                (chat_id,),
            ).fetchall()

    def get_chat_message(self, chat_id: str, message_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT * FROM chat_messages
                WHERE chat_id = ? AND id = ?
                """,
                (chat_id, message_id),
            ).fetchone()

    def delete_chat_messages_from(self, chat_id: str, message_id: str) -> int:
        with self.connect() as conn:
            target = conn.execute(
                """
                SELECT id, created_at
                FROM chat_messages
                WHERE chat_id = ? AND id = ?
                """,
                (chat_id, message_id),
            ).fetchone()
            if not target:
                return 0

            rows = conn.execute(
                """
                SELECT id
                FROM chat_messages
                WHERE chat_id = ? AND created_at >= ?
                ORDER BY created_at ASC
                """,
                (chat_id, target["created_at"]),
            ).fetchall()
            ids = [row["id"] for row in rows]
            if not ids:
                return 0

            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"DELETE FROM chat_messages WHERE id IN ({placeholders})",
                ids,
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), chat_id),
            )
            return len(ids)

    def delete_chat_session(self, chat_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chat_sessions WHERE id = ?", (chat_id,))

    def get_documents_by_upload_id(self, upload_id: str) -> list[sqlite3.Row]:
        if not upload_id:
            return []
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM documents WHERE file_path LIKE ?",
                (f"%{upload_id}%",),
            ).fetchall()

    def get_upload_ocr_stats(self, upload_id: str) -> dict[str, int]:
        stats = {
            "pdf_files_scanned": 0,
            "images_found": 0,
            "image_pages_found": 0,
            "image_pages_processed": 0,
            "image_pages_added_to_rag": 0,
            "image_pages_not_added_to_rag": 0,
        }
        if not upload_id:
            return stats
        pattern = f"%{upload_id}%"
        with self.connect() as conn:
            row_pdf = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM documents
                WHERE file_path LIKE ?
                  AND LOWER(file_ext) = 'pdf'
                """,
                (pattern,),
            ).fetchone()
            stats["pdf_files_scanned"] = int((row_pdf["c"] if row_pdf else 0) or 0)

            row_added = conn.execute(
                """
                SELECT COUNT(DISTINCT c.document_id || ':' || COALESCE(c.anchor_page, -1)) AS c
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.file_path LIKE ?
                  AND c.anchor_type = 'image'
                """,
                (pattern,),
            ).fetchone()
            image_pages_added = int((row_added["c"] if row_added else 0) or 0)
            stats["image_pages_added_to_rag"] = image_pages_added

            # Persisted chunks currently retain only image-source pages that made it into RAG.
            # Use these values as stable fallback stats when no active indexing job is being polled.
            stats["images_found"] = image_pages_added
            stats["image_pages_found"] = image_pages_added
            stats["image_pages_processed"] = image_pages_added
            stats["image_pages_not_added_to_rag"] = 0
        return stats

    def search_chunks_in_upload_by_terms(self, upload_id: str, terms: list[str], limit: int = 24) -> list[sqlite3.Row]:
        if not upload_id or not terms:
            return []
        normalized = [str(t).strip().lower() for t in terms if str(t).strip()]
        if not normalized:
            return []
        like_clauses = " OR ".join(["LOWER(c.text) LIKE ?" for _ in normalized])
        params: list[Any] = [f"%{upload_id}%"] + [f"%{t}%" for t in normalized] + [int(limit)]
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT c.*, d.file_path, d.file_name, d.file_ext
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE d.file_path LIKE ?
                  AND ({like_clauses})
                ORDER BY c.anchor_page, c.anchor_paragraph, c.chunk_index
                LIMIT ?
                """,
                params,
            ).fetchall()

    def search_chunks_in_document_by_terms(self, document_id: str, terms: list[str], limit: int = 12) -> list[sqlite3.Row]:
        if not document_id or not terms:
            return []
        normalized = [str(t).strip().lower() for t in terms if str(t).strip()]
        if not normalized:
            return []
        like_clauses = " OR ".join(["LOWER(c.text) LIKE ?" for _ in normalized])
        params: list[Any] = [document_id] + [f"%{t}%" for t in normalized] + [int(limit)]
        with self.connect() as conn:
            return conn.execute(
                f"""
                SELECT c.*, d.file_path, d.file_name, d.file_ext
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id = ?
                  AND ({like_clauses})
                ORDER BY c.anchor_page, c.anchor_paragraph, c.chunk_index
                LIMIT ?
                """,
                params,
            ).fetchall()

    @staticmethod
    def _fts_escape(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", (query or "").lower())
        if not tokens:
            return ""
        uniq_tokens = list(dict.fromkeys(tokens))
        return " OR ".join(f'"{token}"' for token in uniq_tokens)

    @staticmethod
    def _ensure_chat_session_upload_id_column(conn: sqlite3.Connection) -> None:
        cols = conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
        names = {row[1] for row in cols}
        if "upload_id" not in names:
            conn.execute("ALTER TABLE chat_sessions ADD COLUMN upload_id TEXT")
