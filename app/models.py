from typing import Literal

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    folder_path: str
    mode: str | None = None


class IndexResponse(BaseModel):
    job_id: str | None = None
    upload_id: str | None = None


class OcrStatsResponse(BaseModel):
    pdf_files_scanned: int = 0
    images_found: int = 0
    image_pages_found: int = 0
    image_pages_processed: int = 0
    image_pages_added_to_rag: int = 0
    image_pages_not_added_to_rag: int = 0


class IndexStatusResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = Field(ge=0.0, le=100.0)
    files_total: int
    files_processed: int
    chunks_indexed: int = 0
    warnings: list[str]
    errors: list[str]
    detail: str | None = None
    ocr: OcrStatsResponse = Field(default_factory=OcrStatsResponse)


class ChatRequest(BaseModel):
    message: str
    mode: Literal["general", "fast", "moderate", "accurate"] = "accurate"
    upload_id: str | None = None
    chat_id: str | None = None


class Citation(BaseModel):
    citation_id: str
    chunk_id: str
    file_path: str
    anchor_type: str
    anchor_page: int | None = None
    anchor_section: str | None = None
    anchor_paragraph: int | None = None
    anchor_row: int | None = None
    quoted_snippet: str


class ChatResponse(BaseModel):
    answer_markdown: str
    citations: list[Citation]
    intent_used: str


class ChunkResponse(BaseModel):
    id: str
    document_id: str
    file_path: str
    file_name: str
    file_ext: str
    text: str
    preview: str
    anchor_type: str
    anchor_page: int | None
    anchor_section: str | None
    anchor_paragraph: int | None
    anchor_row: int | None
    start_char: int | None
    end_char: int | None
    file_size_bytes: int | None = None


class ClearIndexResponse(BaseModel):
    ok: bool


class CreateChatRequest(BaseModel):
    title: str | None = None
    upload_id: str | None = None


class RenameChatRequest(BaseModel):
    title: str


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    upload_id: str | None = None
    created_at: str
    updated_at: str


class SaveMessageRequest(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    citations: list[dict] = []


class ChatMessageResponse(BaseModel):
    id: str
    chat_id: str
    role: str
    content: str
    citations: list[dict]
    created_at: str


class DeleteUploadFileRequest(BaseModel):
    file_name: str


class DeleteUploadFileResponse(BaseModel):
    ok: bool
    removed_documents: int = 0


class UploadStatusResponse(BaseModel):
    upload_id: str
    has_indexed_content: bool
    document_count: int = 0
    chunk_count: int = 0
    ocr: OcrStatsResponse = Field(default_factory=OcrStatsResponse)


class UploadFileItem(BaseModel):
    name: str
    status: str
    type: str
    chunk_count: int = 0
    file_size_bytes: int | None = None
    file_path: str


class UploadFilesResponse(BaseModel):
    upload_id: str
    files: list[UploadFileItem]
