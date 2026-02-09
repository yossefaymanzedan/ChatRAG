import csv
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document as DocxDocument
from pptx import Presentation
from pypdf import PdfReader


SUPPORTED_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".csv"}


@dataclass
class ParseResult:
    chunks: list[dict]
    warnings: list[str]


def token_count(text: str) -> int:
    return len(text.split())


def preview_text(text: str, max_len: int = 280) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def split_blocks(text: str) -> list[str]:
    raw = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    if raw:
        return raw
    line_blocks = [line.strip() for line in text.splitlines() if line.strip()]
    return line_blocks


def _looks_like_toc_lines(lines: list[str]) -> bool:
    toc_like_lines = 0
    for ln in lines[:24]:
        has_dot_leader = ("..." in ln) or (" . " in ln)
        ends_with_page = bool(re.search(r"\b\d{1,4}\s*$", ln))
        starts_with_index = bool(re.match(r"^\s*(chapter|part|appendix|\d+(\.\d+){0,3})\b", ln.lower()))
        if ends_with_page and (has_dot_leader or starts_with_index):
            toc_like_lines += 1
    return toc_like_lines >= 3


def _looks_like_front_matter_heading(text: str) -> bool:
    heading = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not heading:
        return False
    prefixes = (
        "acknowledgment",
        "acknowledgement",
        "preface",
        "foreword",
        "copyright",
        "about the author",
        "about this book",
        "list of figures",
        "list of tables",
        "table of contents",
        "contents",
    )
    return any(heading.startswith(p) for p in prefixes)


def _looks_like_acknowledgment_list(text: str) -> bool:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 8:
        return False
    bullet_lines = [ln for ln in lines if ln.startswith(("•", "-", "*"))]
    if len(bullet_lines) < 6:
        return False
    low = re.sub(r"\s+", " ", (text or "").lower())
    signal_terms = ("typo", "suggest", "correction", "chapter", "thanks", "thank")
    signal_hits = sum(1 for t in signal_terms if t in low)
    return signal_hits >= 2


def looks_like_toc_block(text: str) -> bool:
    low = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not low:
        return False

    if "table of contents" in low or low.startswith("contents"):
        return True

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False

    return _looks_like_toc_lines(lines)


def _detect_main_content_start(page_texts: list[tuple[int, str]], scan_pages: int) -> int | None:
    for page_num, text in page_texts[: max(1, scan_pages)]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        head = " ".join(lines[:5]).lower()
        words = token_count(text)
        chapter_like = bool(
            re.search(r"\b(chapter\s+\d+|part\s+[ivx0-9]+|introduction)\b", head)
        )
        if chapter_like and words >= 80 and not _looks_like_toc_lines(lines):
            return page_num
    return None


def _is_front_matter_block(block: str, page_num: int, start_page: int | None, scan_pages: int) -> bool:
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return False
    low = re.sub(r"\s+", " ", block.strip().lower())

    if looks_like_toc_block(block):
        return True

    front_window = max(1, scan_pages)
    if start_page is not None:
        front_window = min(front_window, max(1, start_page - 1))
    in_front_range = page_num <= front_window
    short_block = token_count(block) <= 220

    if in_front_range and _looks_like_front_matter_heading(low):
        return True

    if in_front_range and short_block and re.match(r"^(isbn|copyright|all rights reserved)\b", low):
        return True

    if in_front_range and _looks_like_acknowledgment_list(block):
        return True

    return False


def _is_front_matter_page(text: str, page_num: int, start_page: int | None, scan_pages: int) -> bool:
    front_window = max(1, scan_pages)
    if start_page is not None:
        front_window = min(front_window, max(1, start_page - 1))
    if page_num > front_window:
        return False

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return False
    head = " ".join(lines[:8])
    if _looks_like_front_matter_heading(head):
        return True
    if _looks_like_toc_lines(lines):
        return True
    if _looks_like_acknowledgment_list(text):
        return True
    return False


def parse_file(
    path: Path,
    pdf_text_threshold: int = 40,
    ignore_front_matter: bool = True,
    front_matter_scan_pages: int = 24,
) -> ParseResult:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(
            path,
            pdf_text_threshold,
            ignore_front_matter=ignore_front_matter,
            front_matter_scan_pages=front_matter_scan_pages,
        )
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".pptx":
        return parse_pptx(path)
    if ext == ".md":
        return parse_markdown(path)
    if ext == ".txt":
        return parse_txt(path)
    if ext == ".csv":
        return parse_csv(path)
    return ParseResult(chunks=[], warnings=[f"Unsupported file extension: {ext}"])


def parse_pdf(
    path: Path,
    threshold: int,
    ignore_front_matter: bool = True,
    front_matter_scan_pages: int = 24,
) -> ParseResult:
    chunks: list[dict] = []
    warnings: list[str] = []
    idx = 0
    skipped_toc_blocks = 0
    skipped_front_matter_blocks = 0
    skipped_front_matter_pages = 0
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        return ParseResult(chunks=[], warnings=[f"{path.name}: failed to open PDF ({exc})"])

    page_texts: list[tuple[int, str]] = []
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            warnings.append(f"{path.name}: page {page_num} parse error ({exc})")
            continue
        text = text.strip()
        if len(text) < threshold:
            warnings.append(f"{path.name}: page {page_num} appears scanned/low-text; OCR may be needed")
            continue
        page_texts.append((page_num, text))

    start_page = None
    if ignore_front_matter:
        start_page = _detect_main_content_start(page_texts, front_matter_scan_pages)

    for page_num, text in page_texts:
        if ignore_front_matter and _is_front_matter_page(
            text,
            page_num=page_num,
            start_page=start_page,
            scan_pages=front_matter_scan_pages,
        ):
            skipped_front_matter_pages += 1
            continue
        blocks = split_blocks(text)
        for para_idx, block in enumerate(blocks, start=1):
            if looks_like_toc_block(block):
                skipped_toc_blocks += 1
                continue
            if ignore_front_matter and _is_front_matter_block(
                block,
                page_num=page_num,
                start_page=start_page,
                scan_pages=front_matter_scan_pages,
            ):
                skipped_front_matter_blocks += 1
                continue
            chunks.append(
                {
                    "chunk_index": idx,
                    "text": block,
                    "token_count": token_count(block),
                    "anchor_type": "pdf_page",
                    "anchor_page": page_num,
                    "anchor_section": None,
                    "anchor_paragraph": para_idx,
                    "anchor_row": None,
                    "start_char": None,
                    "end_char": None,
                    "preview": preview_text(block),
                }
            )
            idx += 1

    if skipped_toc_blocks:
        warnings.append(f"{path.name}: skipped {skipped_toc_blocks} TOC-like block(s)")
    if skipped_front_matter_blocks:
        warnings.append(f"{path.name}: skipped {skipped_front_matter_blocks} front-matter block(s)")
    if skipped_front_matter_pages:
        warnings.append(f"{path.name}: skipped {skipped_front_matter_pages} front-matter page(s)")
    if ignore_front_matter and start_page is not None:
        warnings.append(f"{path.name}: detected main content start around page {start_page}")

    return ParseResult(chunks=chunks, warnings=warnings)


def parse_docx(path: Path) -> ParseResult:
    doc = DocxDocument(str(path))
    para_records: list[tuple[int, str, str | None]] = []
    nearest_heading: str | None = None

    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            nearest_heading = text
            continue
        para_records.append((i, text, nearest_heading))

    merged: list[tuple[int, str, str | None]] = []
    i = 0
    while i < len(para_records):
        start_idx, text, section = para_records[i]
        while len(text) < 80 and i + 1 < len(para_records):
            i += 1
            _, nxt_text, _ = para_records[i]
            text = f"{text}\n{nxt_text}"
            if len(text) >= 160:
                break
        merged.append((start_idx, text, section))
        i += 1

    chunks: list[dict] = []
    for chunk_idx, (paragraph_index, text, section) in enumerate(merged):
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "text": text,
                "token_count": token_count(text),
                "anchor_type": "docx_paragraph",
                "anchor_page": None,
                "anchor_section": section,
                "anchor_paragraph": paragraph_index,
                "anchor_row": None,
                "start_char": None,
                "end_char": None,
                "preview": preview_text(text),
            }
        )

    return ParseResult(chunks=chunks, warnings=[])


def parse_pptx(path: Path) -> ParseResult:
    prs = Presentation(str(path))
    chunks: list[dict] = []
    chunk_idx = 0
    for slide_num, slide in enumerate(prs.slides, start=1):
        block_idx = 1
        for shape in slide.shapes:
            text = getattr(shape, "text", "") or ""
            text = text.strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "text": text,
                    "token_count": token_count(text),
                    "anchor_type": "ppt_slide",
                    "anchor_page": slide_num,
                    "anchor_section": f"Slide {slide_num}",
                    "anchor_paragraph": block_idx,
                    "anchor_row": None,
                    "start_char": None,
                    "end_char": None,
                    "preview": preview_text(text),
                }
            )
            chunk_idx += 1
            block_idx += 1
    return ParseResult(chunks=chunks, warnings=[])


def parse_markdown(path: Path) -> ParseResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    chunks: list[dict] = []
    current_heading = "Document"
    block: list[str] = []
    block_index = 0

    def flush() -> None:
        nonlocal block, block_index
        content = "\n".join(block).strip()
        if not content:
            block = []
            return
        chunks.append(
            {
                "chunk_index": len(chunks),
                "text": content,
                "token_count": token_count(content),
                "anchor_type": "md_heading",
                "anchor_page": None,
                "anchor_section": current_heading,
                "anchor_paragraph": block_index,
                "anchor_row": None,
                "start_char": None,
                "end_char": None,
                "preview": preview_text(content),
            }
        )
        block_index += 1
        block = []

    for line in lines:
        heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if heading:
            flush()
            current_heading = heading.group(2).strip()
            block_index = 0
            continue
        if line.strip() == "":
            flush()
        else:
            block.append(line)
    flush()

    return ParseResult(chunks=chunks, warnings=[])


def parse_txt(path: Path) -> ParseResult:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = split_blocks(text)
    chunks = []
    for idx, block in enumerate(blocks):
        chunks.append(
            {
                "chunk_index": idx,
                "text": block,
                "token_count": token_count(block),
                "anchor_type": "txt_block",
                "anchor_page": None,
                "anchor_section": None,
                "anchor_paragraph": idx,
                "anchor_row": None,
                "start_char": None,
                "end_char": None,
                "preview": preview_text(block),
            }
        )

    return ParseResult(chunks=chunks, warnings=[])


def parse_csv(path: Path) -> ParseResult:
    chunks: list[dict] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        headers_str = ", ".join(headers)
        for row_idx, row in enumerate(reader, start=1):
            pairs = " ".join(f"{k}={row.get(k, '')}" for k in headers)
            text = f"Headers: {headers_str}\nRow {row_idx}: {pairs}".strip()
            chunks.append(
                {
                    "chunk_index": row_idx - 1,
                    "text": text,
                    "token_count": token_count(text),
                    "anchor_type": "csv_row",
                    "anchor_page": None,
                    "anchor_section": None,
                    "anchor_paragraph": None,
                    "anchor_row": row_idx,
                    "start_char": None,
                    "end_char": None,
                    "preview": preview_text(text),
                }
            )

    return ParseResult(chunks=chunks, warnings=[])
