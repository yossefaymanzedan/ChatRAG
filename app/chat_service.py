from __future__ import annotations

import json
import re
import logging
from typing import Any
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.database import Database
from app.embeddings import EmbeddingService
from app.llm import DeepSeekClient
from app.models import Citation
from app.retrieval import HybridRetriever, RetrievedChunk


def _build_token_logger() -> logging.Logger:
    logger = logging.getLogger("chatrag.tokens")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    log_path = Path(".rag") / "token_logs.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


TOKEN_LOGGER = _build_token_logger()


def detect_intent(question: str) -> str:
    # Intents are inferred by the model during answer generation.
    return "default"


class ChatService:
    def __init__(
        self,
        db: Database,
        embeddings: EmbeddingService,
        retriever: HybridRetriever,
        llm: DeepSeekClient,
    ) -> None:
        self.db = db
        self.embeddings = embeddings
        self.retriever = retriever
        self.llm = llm

    def chat(self, message: str, mode: str, upload_id: str | None = None, chat_id: str | None = None) -> dict[str, Any]:
        call_id = str(datetime.utcnow().timestamp()).replace(".", "")[-12:]
        if mode == "general":
            history_text = self._build_recent_chat_history_text(chat_id, max_messages=80)
            system_prompt = self._build_system_prompt(mode, "General mode (file retrieval disabled).", json_output=True)
            user_prompt = (
                f"Question: {message}\n\n"
                f"Recent conversation:\n{history_text}\n\n"
                "Return JSON now."
            )
            self._log_token_metrics(
                event="chat_request",
                call_id=call_id,
                mode=mode,
                upload_id=upload_id,
                chat_id=chat_id,
                data={
                    "input_tokens_est": self._estimate_tokens(message),
                    "retrieval_query_tokens_est": 0,
                    "context_tokens_est": 0,
                    "history_tokens_est": self._estimate_tokens(history_text),
                    "system_prompt_tokens_est": self._estimate_tokens(system_prompt),
                    "user_prompt_tokens_est": self._estimate_tokens(user_prompt),
                    "prompt_total_tokens_est": self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt),
                },
            )
            try:
                llm_json = self.llm.chat_json(system_prompt, user_prompt)
            except Exception as exc:
                raise RuntimeError(
                    f"Local LLM request failed. Ensure Ollama is running and model is pulled. Details: {exc}"
                ) from exc
            answer_markdown = str(llm_json.get("answer_markdown") or llm_json.get("answer") or "").strip()
            if not answer_markdown:
                answer_markdown = "I could not generate a response."
            self._log_token_metrics(
                event="chat_response",
                call_id=call_id,
                mode=mode,
                upload_id=upload_id,
                chat_id=chat_id,
                data={
                    "output_tokens_est": self._estimate_tokens(answer_markdown),
                    "output_chars": len(answer_markdown),
                    "citation_count": 0,
                    "total_tokens_est": (
                        self._estimate_tokens(system_prompt)
                        + self._estimate_tokens(user_prompt)
                        + self._estimate_tokens(answer_markdown)
                    ),
                },
            )
            return {
                "answer_markdown": answer_markdown,
                "citations": [],
                "intent_used": str(llm_json.get("intent_used", "general")),
            }

        intent = detect_intent(message)
        retrieval_message = self._expand_followup_query(message, chat_id)
        docs = self.db.get_documents_by_upload_id(upload_id) if upload_id else []
        requested_docs_current = self._find_message_referenced_docs(message, docs)
        scope = self._infer_scope_from_context(message, chat_id, docs)
        requested_docs = self._find_message_referenced_docs(retrieval_message, docs)
        # Only force specific file scope when CURRENT user message itself references file scope.
        if requested_docs_current and scope.get("scope") == "specific" and scope.get("file_name"):
            matched = self._match_doc_by_name(str(scope.get("file_name")), docs)
            if matched is not None:
                requested_docs = [matched]
                retrieval_message = f"{retrieval_message}\nScope file: {matched['file_name']}"
        clarification = self._build_scope_clarification_prompt(retrieval_message, docs, requested_docs, scope)
        if clarification:
            return {
                "answer_markdown": clarification,
                "citations": [],
                "intent_used": intent,
            }
        retrieved, max_score, _trace = self._retrieve_with_trace(retrieval_message, mode, upload_id)
        retrieved = self._filter_by_upload_scope(retrieved, upload_id)
        if mode == "fast":
            retrieved = self._boost_requested_docs_for_fast(message, docs, retrieved)
        retrieved = self._ensure_retrieval_coverage(retrieval_message, retrieved)
        retrieved = self._rescue_challenge_retrieval(retrieved, retrieval_message, upload_id)
        if mode == "fast":
            retrieved = self._fast_last_chance_retrieval(retrieval_message, upload_id, retrieved)
        if mode in {"accurate", "moderate"}:
            retrieved, _dig_meta = self._accurate_deep_dig_if_needed(retrieval_message, upload_id, retrieved, max_score, mode)
        retrieved = self._strip_doc_summary_chunks(retrieved)
        retrieved = self._strip_toc_chunks(retrieved)
        retrieved = self._apply_explicit_file_scope(message, requested_docs_current, retrieved)
        if mode in {"accurate", "moderate"} and not retrieved:
            retrieved = self._strip_doc_summary_chunks(
                self._accurate_per_file_react_fallback(retrieval_message, upload_id)
            )
            retrieved = self._strip_toc_chunks(retrieved)
            retrieved = self._apply_explicit_file_scope(message, requested_docs_current, retrieved)
        if mode in {"accurate", "moderate"} and not retrieved and requested_docs_current:
            retrieved = self._exhaustive_requested_doc_scan(message, requested_docs_current)

        if not retrieved:
            return {
                "answer_markdown": "Not found in indexed docs.",
                "citations": [],
                "intent_used": intent,
            }

        chunks = [item.chunk for item in retrieved]
        citation_map, context = self._build_context_for_mode(chunks, mode)
        if intent == "challenge_count":
            extracted = self._extract_challenges_from_citations(citation_map)
            if extracted:
                answer, citation_ids = self._build_challenge_count_answer(extracted)
                citations = self._citations_from_chunks([citation_map[c] for c in citation_ids], citation_ids)
                return {
                    "answer_markdown": answer,
                    "citations": citations,
                    "intent_used": intent,
                }

        if mode == "fast" and max_score < self._not_found_threshold_for_mode(mode) and len(chunks) <= 2:
            closest = []
            for i, chunk in enumerate(chunks[:3], start=1):
                closest.append(f"- Closest {i}: `{chunk['file_name']}` ({self._chunk_location(chunk)}): {chunk['preview']}")
            return {
                "answer_markdown": "Not found in indexed docs.\n\n" + "\n".join(closest),
                "citations": self._citations_from_chunks(chunks[:3], [f"C{i}" for i in range(1, min(3, len(chunks)) + 1)]),
                "intent_used": intent,
            }

        file_inventory = self._build_file_inventory_text(upload_id)
        system_prompt = self._build_system_prompt(mode, file_inventory, json_output=True)

        user_prompt = (
            f"Intent: {intent}\n"
            f"Question: {message}\n\n"
            + (f"Resolved follow-up context: {retrieval_message}\n\n" if retrieval_message != message else "")
            + (f"Current indexed files:\n{file_inventory}\n\n")
            + (
            f"Citations context:\n{context}\n\n"
            "Respond in JSON now."
            )
        )
        self._log_token_metrics(
            event="chat_request",
            call_id=call_id,
            mode=mode,
            upload_id=upload_id,
            chat_id=chat_id,
            data={
                "input_tokens_est": self._estimate_tokens(message),
                "retrieval_query_tokens_est": self._estimate_tokens(retrieval_message),
                "context_tokens_est": self._estimate_tokens(context),
                "system_prompt_tokens_est": self._estimate_tokens(system_prompt),
                "user_prompt_tokens_est": self._estimate_tokens(user_prompt),
                "prompt_total_tokens_est": self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt),
                "context_chars": len(context),
                "context_chunks": len(citation_map),
            },
        )

        try:
            llm_json = self.llm.chat_json(system_prompt, user_prompt)
        except Exception as exc:
            raise RuntimeError(
                f"Local LLM request failed. Ensure Ollama is running and model is pulled. Details: {exc}"
            ) from exc

        answer_markdown = str(llm_json.get("answer_markdown", "")).strip() or "Not found in indexed docs."
        raw_cits = llm_json.get("citations", [])
        if isinstance(raw_cits, str):
            try:
                raw_cits = json.loads(raw_cits)
            except Exception:
                raw_cits = re.findall(r"C\d+", raw_cits)

        citation_ids = [c for c in raw_cits if isinstance(c, str) and c in citation_map]
        citation_ids = self._compress_citation_ids(citation_ids, citation_map, mode)
        citation_ids = self._ensure_citation_coverage(citation_ids, citation_map, retrieval_message, mode)
        if not citation_ids:
            citation_ids = self._fallback_citation_ids(citation_map, retrieval_message, mode)

        citations = self._citations_from_chunks([citation_map[c] for c in citation_ids], citation_ids)
        self._log_token_metrics(
            event="chat_response",
            call_id=call_id,
            mode=mode,
            upload_id=upload_id,
            chat_id=chat_id,
            data={
                "output_tokens_est": self._estimate_tokens(answer_markdown),
                "output_chars": len(answer_markdown),
                "citation_count": len(citations),
                "total_tokens_est": (
                    self._estimate_tokens(system_prompt)
                    + self._estimate_tokens(user_prompt)
                    + self._estimate_tokens(answer_markdown)
                ),
            },
        )

        return {
            "answer_markdown": answer_markdown,
            "citations": citations,
            "intent_used": str(llm_json.get("intent_used", intent)),
        }

    def chat_stream(self, message: str, mode: str, upload_id: str | None = None, chat_id: str | None = None):
        call_id = str(datetime.utcnow().timestamp()).replace(".", "")[-12:]
        if mode == "general":
            history_text = self._build_recent_chat_history_text(chat_id, max_messages=80)
            system_prompt = self._build_system_prompt(mode, "General mode (file retrieval disabled).", json_output=False)
            user_prompt = (
                f"Question: {message}\n\n"
                f"Recent conversation:\n{history_text}\n\n"
                "Respond with markdown only."
            )
            self._log_token_metrics(
                event="chat_stream_request",
                call_id=call_id,
                mode=mode,
                upload_id=upload_id,
                chat_id=chat_id,
                data={
                    "input_tokens_est": self._estimate_tokens(message),
                    "retrieval_query_tokens_est": 0,
                    "context_tokens_est": 0,
                    "history_tokens_est": self._estimate_tokens(history_text),
                    "system_prompt_tokens_est": self._estimate_tokens(system_prompt),
                    "user_prompt_tokens_est": self._estimate_tokens(user_prompt),
                    "prompt_total_tokens_est": self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt),
                },
            )

            answer = ""
            try:
                for delta in self.llm.chat_stream_markdown(system_prompt, user_prompt):
                    answer += delta
                    yield {"event": "token", "data": {"delta": delta}}
            except Exception as exc:
                raise RuntimeError(
                    f"Local LLM stream failed. Ensure Ollama is running and model is pulled. Details: {exc}"
                ) from exc

            self._log_token_metrics(
                event="chat_stream_response",
                call_id=call_id,
                mode=mode,
                upload_id=upload_id,
                chat_id=chat_id,
                data={
                    "output_tokens_est": self._estimate_tokens(answer),
                    "output_chars": len(answer),
                    "citation_count": 0,
                    "total_tokens_est": (
                        self._estimate_tokens(system_prompt)
                        + self._estimate_tokens(user_prompt)
                        + self._estimate_tokens(answer)
                    ),
                },
            )
            yield {
                "event": "done",
                "data": {
                    "answer_markdown": answer.strip() or "I could not generate a response.",
                    "citations": [],
                    "intent_used": "general",
                },
            }
            return

        intent = detect_intent(message)
        retrieval_message = self._expand_followup_query(message, chat_id)
        docs = self.db.get_documents_by_upload_id(upload_id) if upload_id else []
        requested_docs_current = self._find_message_referenced_docs(message, docs)
        scope = self._infer_scope_from_context(message, chat_id, docs)
        requested_docs = self._find_message_referenced_docs(retrieval_message, docs)
        # Only force specific file scope when CURRENT user message itself references file scope.
        if requested_docs_current and scope.get("scope") == "specific" and scope.get("file_name"):
            matched = self._match_doc_by_name(str(scope.get("file_name")), docs)
            if matched is not None:
                requested_docs = [matched]
                retrieval_message = f"{retrieval_message}\nScope file: {matched['file_name']}"
        clarification = self._build_scope_clarification_prompt(retrieval_message, docs, requested_docs, scope)
        if clarification:
            yield {
                "event": "done",
                "data": {
                    "answer_markdown": clarification,
                    "citations": [],
                    "intent_used": intent,
                },
            }
            return
        if mode in {"accurate", "moderate"}:
            retrieved = []
            max_score = 0.0
            trace = {"planner": "react_graph", "queries": [], "probes": []}
            for step in self._accurate_langgraph_retrieval_stream(retrieval_message, upload_id):
                if step.get("type") == "detail":
                    yield {"event": "status_detail", "data": {"line": str(step.get("line", ""))}}
                elif step.get("type") == "result":
                    retrieved = step.get("retrieved", []) or []
                    max_score = float(step.get("max_score", 0.0) or 0.0)
                    trace = step.get("trace", trace) or trace
        else:
            retrieved, max_score, trace = self._retrieve_with_trace(retrieval_message, mode, upload_id)
        raw_hits = len(retrieved)
        retrieved = self._filter_by_upload_scope(retrieved, upload_id)
        if mode == "fast":
            retrieved = self._boost_requested_docs_for_fast(message, docs, retrieved)
        retrieved = self._ensure_retrieval_coverage(retrieval_message, retrieved)
        pre_rescue_hits = len(retrieved)
        retrieved = self._rescue_challenge_retrieval(retrieved, retrieval_message, upload_id)
        if mode == "fast":
            retrieved = self._fast_last_chance_retrieval(retrieval_message, upload_id, retrieved)
        dig_meta: dict[str, Any] = {}
        if mode in {"accurate", "moderate"}:
            retrieved, dig_meta = self._accurate_deep_dig_if_needed(message, upload_id, retrieved, max_score, mode)
        retrieved = self._strip_doc_summary_chunks(retrieved)
        retrieved = self._strip_toc_chunks(retrieved)
        retrieved = self._apply_explicit_file_scope(message, requested_docs_current, retrieved)
        if mode in {"accurate", "moderate"} and not retrieved:
            retrieved = self._strip_doc_summary_chunks(
                self._accurate_per_file_react_fallback(retrieval_message, upload_id)
            )
            retrieved = self._strip_toc_chunks(retrieved)
            retrieved = self._apply_explicit_file_scope(message, requested_docs_current, retrieved)
        if mode in {"accurate", "moderate"} and not retrieved and requested_docs_current:
            retrieved = self._exhaustive_requested_doc_scan(message, requested_docs_current)
        final_hits = len(retrieved)
        yield {
            "event": "status",
            "data": {
                "stage": "retrieval",
                "mode": mode,
                "planner": trace.get("planner", "fast"),
                "queries": trace.get("queries", []),
                "hits": final_hits,
            },
        }
        for line in self._build_react_notes(
            message=message,
            retrieval_message=retrieval_message,
            mode=mode,
            max_score=max_score,
            raw_hits=raw_hits,
            pre_rescue_hits=pre_rescue_hits,
            final_hits=final_hits,
            trace=trace,
        ):
            yield {"event": "status_detail", "data": {"line": line}}
        if dig_meta.get("triggered"):
            files = dig_meta.get("files", [])
            if files:
                yield {"event": "status_detail", "data": {"line": f"Deep dig triggered: probing {len(files)} likely file(s)."}}
                for f in files[:6]:
                    yield {"event": "status_detail", "data": {"line": f"Digging in file: {f}"}}
            for line in dig_meta.get("notes", [])[:30]:
                yield {"event": "status_detail", "data": {"line": line}}

        if not retrieved:
            yield {
                "event": "done",
                "data": {
                    "answer_markdown": "Not found in indexed docs.",
                    "citations": [],
                    "intent_used": intent,
                },
            }
            return

        chunks = [item.chunk for item in retrieved]
        citation_map, context = self._build_context_for_mode(chunks, mode)
        if intent == "challenge_count":
            extracted = self._extract_challenges_from_citations(citation_map)
            if extracted:
                answer, citation_ids = self._build_challenge_count_answer(extracted)
                citations = self._citations_from_chunks([citation_map[c] for c in citation_ids], citation_ids)
                yield {
                    "event": "done",
                    "data": {
                        "answer_markdown": answer,
                        "citations": citations,
                        "intent_used": intent,
                    },
                }
                return

        if mode == "fast" and max_score < self._not_found_threshold_for_mode(mode) and len(chunks) <= 2:
            fallback = "Not found in indexed docs.\n\n"
            for i, chunk in enumerate(chunks[:3], start=1):
                fallback += f"- Closest {i}: `{chunk['file_name']}` ({self._chunk_location(chunk)}): {chunk['preview']}\n"
            yield {
                "event": "done",
                "data": {
                    "answer_markdown": fallback.strip(),
                    "citations": self._citations_from_chunks(
                        chunks[:3], [f"C{i}" for i in range(1, min(3, len(chunks)) + 1)]
                    ),
                    "intent_used": intent,
                },
            }
            return

        file_inventory = self._build_file_inventory_text(upload_id)
        system_prompt = self._build_system_prompt(mode, file_inventory, json_output=False)
        user_prompt = (
            f"Intent: {intent}\n"
            f"Question: {message}\n\n"
            + (f"Resolved follow-up context: {retrieval_message}\n\n" if retrieval_message != message else "")
            + (f"Current indexed files:\n{file_inventory}\n\n")
            + (
            f"Citations context:\n{context}\n\n"
            "Respond with markdown only."
            )
        )
        self._log_token_metrics(
            event="chat_stream_request",
            call_id=call_id,
            mode=mode,
            upload_id=upload_id,
            chat_id=chat_id,
            data={
                "input_tokens_est": self._estimate_tokens(message),
                "retrieval_query_tokens_est": self._estimate_tokens(retrieval_message),
                "context_tokens_est": self._estimate_tokens(context),
                "system_prompt_tokens_est": self._estimate_tokens(system_prompt),
                "user_prompt_tokens_est": self._estimate_tokens(user_prompt),
                "prompt_total_tokens_est": self._estimate_tokens(system_prompt) + self._estimate_tokens(user_prompt),
                "context_chars": len(context),
                "context_chunks": len(citation_map),
            },
        )

        answer = ""
        try:
            for delta in self.llm.chat_stream_markdown(system_prompt, user_prompt):
                answer += delta
                yield {"event": "token", "data": {"delta": delta}}
        except Exception as exc:
            raise RuntimeError(
                f"Local LLM stream failed. Ensure Ollama is running and model is pulled. Details: {exc}"
            ) from exc

        cited_ids = [cid for cid in re.findall(r"C\d+", answer) if cid in citation_map]
        deduped = list(dict.fromkeys(cited_ids))
        deduped = self._compress_citation_ids(deduped, citation_map, mode)
        deduped = self._ensure_citation_coverage(deduped, citation_map, retrieval_message, mode)
        if not deduped:
            deduped = self._fallback_citation_ids(citation_map, retrieval_message, mode)

        citations = self._citations_from_chunks([citation_map[c] for c in deduped], deduped)
        self._log_token_metrics(
            event="chat_stream_response",
            call_id=call_id,
            mode=mode,
            upload_id=upload_id,
            chat_id=chat_id,
            data={
                "output_tokens_est": self._estimate_tokens(answer),
                "output_chars": len(answer),
                "citation_count": len(citations),
                "total_tokens_est": (
                    self._estimate_tokens(system_prompt)
                    + self._estimate_tokens(user_prompt)
                    + self._estimate_tokens(answer)
                ),
            },
        )
        yield {
            "event": "done",
            "data": {
                "answer_markdown": answer.strip() or "Not found in indexed docs.",
                "citations": citations,
                "intent_used": intent,
            },
        }

    @staticmethod
    def _not_found_threshold_for_mode(mode: str) -> float:
        if mode == "fast":
            return min(float(settings.hard_not_found_threshold), 0.08)
        return float(settings.hard_not_found_threshold)

    def _boost_requested_docs_for_fast(self, message: str, docs: list[Any], retrieved):
        if not retrieved:
            return retrieved
        requested = self._find_message_referenced_docs(message, docs)
        if not requested:
            return retrieved
        requested_ids = {str(d["id"]) for d in requested}
        boosted = []
        normal = []
        for item in retrieved:
            did = str(item.chunk.get("document_id") or "")
            if did in requested_ids:
                boosted.append(item)
            else:
                normal.append(item)
        out = boosted + normal
        # preserve fast latency/cost profile
        return out[: max(8, len(boosted))]

    def _fast_last_chance_retrieval(self, message: str, upload_id: str | None, retrieved):
        if not upload_id:
            return retrieved
        if len(retrieved) >= 5:
            return retrieved
        terms = self._extract_terms_for_rescue(message)
        if not terms:
            return retrieved

        rows = self.db.search_chunks_in_upload_by_terms(upload_id, terms[:36], limit=42)
        if not rows:
            return retrieved

        out = list(retrieved)
        seen = {str(r.chunk.get("id") or "") for r in out}
        for row in rows:
            chunk = dict(row)
            cid = str(chunk.get("id") or "")
            if not cid or cid in seen:
                continue
            if self._looks_like_toc(chunk):
                continue
            seen.add(cid)
            out.append(RetrievedChunk(chunk=chunk, score=0.035))
            if len(out) >= 12:
                break
        return out

    def _build_context_for_mode(self, chunks: list[dict[str, Any]], mode: str) -> tuple[dict[str, dict[str, Any]], str]:
        citation_map: dict[str, dict[str, Any]] = {}
        context_lines: list[str] = []
        if not chunks:
            return citation_map, ""

        if mode == "accurate":
            # Accurate mode: keep retrieval wide, but constrain prompt context with a relevance-first budget.
            # Hard floor: always include top-N strongest chunks before budget trimming.
            max_chunks = 28
            max_chunk_chars = 1400
            max_total_context_tokens = 14000
            max_total_chars = 70000
            must_keep_top = 10
            total_tokens = 0
            total_chars = 0
            seen_sig: set[str] = set()
            selected: list[dict[str, Any]] = []
            seen_doc_counts: dict[str, int] = {}

            for chunk in chunks:
                sig = "|".join(
                    [
                        str(chunk.get("file_path", "")),
                        str(chunk.get("anchor_page", "")),
                        str(chunk.get("anchor_paragraph", "")),
                        str(chunk.get("anchor_row", "")),
                    ]
                )
                if sig in seen_sig:
                    continue
                did = str(chunk.get("document_id") or "")
                # Soft diversity guard without overriding score order.
                if did and seen_doc_counts.get(did, 0) >= 12:
                    continue
                seen_sig.add(sig)
                if did:
                    seen_doc_counts[did] = seen_doc_counts.get(did, 0) + 1
                selected.append(chunk)
                if len(selected) >= max_chunks:
                    break

            if not selected:
                selected = chunks[:1]

            for i, chunk in enumerate(selected, start=1):
                citation_id = f"C{i}"
                citation_map[citation_id] = chunk
                loc = self._chunk_location(chunk)
                text = self._compact_context_text(str(chunk.get("text", "")), max_chunk_chars)
                line = f"[{citation_id}] {chunk['file_name']} ({loc})\n{text}"
                line_tokens = self._estimate_tokens(line)
                if i > must_keep_top and (
                    (total_tokens + line_tokens) > max_total_context_tokens
                    or (total_chars + len(line)) > max_total_chars
                ) and context_lines:
                    break
                context_lines.append(line)
                total_tokens += line_tokens
                total_chars += len(line)

            return citation_map, "\n\n".join(context_lines)

        if mode == "moderate":
            # Moderate mode: relevance-first with bounded prompt size.
            # Keep higher-quality top chunks first, then add diversity extras.
            max_chunks = 24
            max_chunk_chars = 1300
            max_total_context_tokens = 12000
            max_total_chars = 52000
            total_tokens = 0
            total_chars = 0
            seen_sig: set[str] = set()
            selected: list[dict[str, Any]] = []
            seen_doc_counts: dict[str, int] = {}

            for chunk in chunks:
                sig = "|".join(
                    [
                        str(chunk.get("file_path", "")),
                        str(chunk.get("anchor_page", "")),
                        str(chunk.get("anchor_paragraph", "")),
                        str(chunk.get("anchor_row", "")),
                    ]
                )
                if sig in seen_sig:
                    continue
                did = str(chunk.get("document_id") or "")
                # Soft diversity: avoid one-document dominance, but do not override relevance ordering.
                if did and seen_doc_counts.get(did, 0) >= 10:
                    continue
                seen_sig.add(sig)
                if did:
                    seen_doc_counts[did] = seen_doc_counts.get(did, 0) + 1
                selected.append(chunk)
                if len(selected) >= max_chunks:
                    break

            if not selected:
                selected = chunks[:1]

            for i, chunk in enumerate(selected, start=1):
                citation_id = f"C{i}"
                citation_map[citation_id] = chunk
                loc = self._chunk_location(chunk)
                text = self._compact_context_text(str(chunk.get("text", "")), max_chunk_chars)
                line = f"[{citation_id}] {chunk['file_name']} ({loc})\n{text}"
                line_tokens = self._estimate_tokens(line)
                if ((total_tokens + line_tokens) > max_total_context_tokens or (total_chars + len(line)) > max_total_chars) and context_lines:
                    break
                context_lines.append(line)
                total_tokens += line_tokens
                total_chars += len(line)

            return citation_map, "\n\n".join(context_lines)

        # Fast mode token guard: allow higher context while keeping bounded cost.
        # Target roughly <= 4.5k prompt tokens including system/user wrappers.
        max_chunks = 12
        max_chunk_chars = 2200
        max_total_context_tokens = 3600
        max_total_chars = 18000
        total_tokens = 0
        total_chars = 0
        seen_sig: set[str] = set()
        selected: list[dict[str, Any]] = []
        docs_first: list[dict[str, Any]] = []
        tail: list[dict[str, Any]] = []
        seen_doc: set[str] = set()
        for chunk in chunks:
            did = str(chunk.get("document_id") or "")
            if did and did not in seen_doc:
                seen_doc.add(did)
                docs_first.append(chunk)
            else:
                tail.append(chunk)

        for chunk in docs_first + tail:
            sig = "|".join(
                [
                    str(chunk.get("file_path", "")),
                    str(chunk.get("anchor_page", "")),
                    str(chunk.get("anchor_paragraph", "")),
                    str(chunk.get("anchor_row", "")),
                ]
            )
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            selected.append(chunk)
            if len(selected) >= max_chunks:
                break

        if not selected:
            selected = chunks[:1]

        for i, chunk in enumerate(selected, start=1):
            citation_id = f"C{i}"
            citation_map[citation_id] = chunk
            loc = self._chunk_location(chunk)
            text = self._compact_context_text(str(chunk.get("text", "")), max_chunk_chars)
            line = f"[{citation_id}] {chunk['file_name']} ({loc})\n{text}"
            line_tokens = self._estimate_tokens(line)
            if ((total_tokens + line_tokens) > max_total_context_tokens or (total_chars + len(line)) > max_total_chars) and context_lines:
                break
            context_lines.append(line)
            total_tokens += line_tokens
            total_chars += len(line)

        return citation_map, "\n\n".join(context_lines)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        # Lightweight estimator: word/punct segments with a small overhead factor.
        segments = re.findall(r"\w+|[^\w\s]", str(text), flags=re.UNICODE)
        return int(max(1, round(len(segments) * 1.05)))

    def _log_token_metrics(
        self,
        *,
        event: str,
        call_id: str,
        mode: str,
        upload_id: str | None,
        chat_id: str | None,
        data: dict[str, Any],
    ) -> None:
        payload = {
            "ts_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            "call_id": call_id,
            "mode": mode,
            "upload_id": upload_id,
            "chat_id": chat_id,
            **data,
        }
        try:
            TOKEN_LOGGER.info(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    @staticmethod
    def _compact_context_text(text: str, max_chars: int) -> str:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + "..."

    def _build_system_prompt(self, mode: str, file_inventory: str, *, json_output: bool) -> str:
        if mode == "general":
            out_rule = (
                "Return strict JSON object with keys: answer_markdown, intent_used."
                if json_output
                else "Write markdown only."
            )
            return (
                "You are a helpful assistant in general-chat mode. "
                "Do not claim you searched files or provide fake citations. "
                "Use recent conversation context when relevant. "
                "Be concise, direct, and practical. "
                f"{out_rule}"
            )

        if mode == "fast":
            out_rule = (
                "Return strict JSON object with keys: answer_markdown, citations, intent_used. "
                "citations must be an array of citation IDs such as C1, C2."
                if json_output
                else "Write markdown and cite claims with citation IDs like [C1], [C2]."
            )
            return (
                "You are a grounded RAG assistant. Use only the supplied citations. "
                "Answer directly and concisely. "
                "If evidence is missing, say: Not found in indexed docs. "
                "Do not claim facts not present in citations. "
                "For every factual claim, state the supporting evidence from the provided context and cite it. "
                "Always show where the data came from by attaching citation IDs to claims. "
                "For list/exhaustive requests, include all relevant items present in the supplied citations. "
                "Use as few citations as needed (prefer 1-4). "
                f"{out_rule}\n\n"
                f"FILE_INVENTORY:\n{file_inventory}"
            )

        out_rule = (
            "Return strict JSON object with keys: answer_markdown, citations, intent_used. "
            "citations must be an array of citation IDs such as C1, C2."
            if json_output
            else "Write markdown and cite claims with citation IDs like [C1], [C2]."
        )
        return (
            "You are a grounded RAG assistant. Use only the supplied citations. "
            "Write like a helpful human teammate in natural conversation. "
            "For every factual claim, provide evidence from the supplied context and cite it. "
            "Always show where the data came from by attaching citation IDs to claims. "
            "Do NOT use stiff boilerplate such as 'Based solely on the provided citation'. "
            "Interpret figurative/slang wording by intent (e.g., 'magic engineer' means 'exceptional engineer'). "
            "Start with a direct human answer first, then support with evidence. "
            "If evidence is missing, say clearly and briefly: Not found in indexed docs. "
            "For evaluative questions, infer cautiously from evidence in citations and make criteria explicit. "
            "Keep it concise, direct, and practical. "
            "Use the minimum necessary citations and avoid repeating citations from the same snippet/page. "
            "Prefer 2-6 citations max unless strictly necessary. "
            "For broad questions, synthesize across multiple relevant chunks instead of relying on a single chunk. "
            "Do not force taxonomy labels from user wording. Validate labels against evidence. "
            "If label meaning is ambiguous, present evidence-backed interpretation(s) instead of hard classification. "
            "Prefer describing function/behavior/relationship when that's what citations support. "
            "Do not claim items that are not explicit in the citations. "
            "For count questions, provide an exact number only if the citations clearly enumerate countable items; otherwise say that exact count is not explicit. "
            "For exhaustive/list requests (e.g., starts with 'list all', 'all requirements', 'enumerate'), include ALL matching evidence across relevant files. "
            "If the same requirement appears in multiple files, list each file occurrence explicitly. "
            "Do not narrow to one file unless the current user message explicitly scopes to that file. "
            "You are aware of the current indexed file inventory (count + file names) provided below; "
            "use it to avoid mixing files and to align answers with user-mentioned file names. "
            f"{out_rule}\n\n"
            f"FILE_INVENTORY:\n{file_inventory}"
        )

    def _build_file_inventory_text(self, upload_id: str | None) -> str:
        if not upload_id:
            return "No upload is currently selected."
        docs = self.db.get_documents_by_upload_id(upload_id)
        if not docs:
            return "0 files indexed for current upload."
        names: list[str] = []
        for doc in docs:
            name = str(doc["file_name"] if "file_name" in doc.keys() else "").strip()
            if name:
                names.append(name)
        unique_names = sorted(set(names), key=lambda x: x.lower())
        if not unique_names:
            return "0 files indexed for current upload."

        max_visible = 30
        shown = unique_names[:max_visible]
        lines = [f"{i + 1}. {name}" for i, name in enumerate(shown)]
        if len(unique_names) > max_visible:
            lines.append(f"... (+{len(unique_names) - max_visible} more file(s))")
        return f"{len(unique_names)} indexed file(s)\n" + "\n".join(lines)

    def _build_recent_chat_history_text(self, chat_id: str | None, max_messages: int = 80) -> str:
        if not chat_id:
            return "(none)"
        try:
            rows = self.db.list_chat_messages(chat_id)
        except Exception:
            return "(none)"
        if not rows:
            return "(none)"
        lines: list[str] = []
        # General mode memory excludes file-grounded turns (messages carrying citations),
        # so history stays conversational and does not replay retrieved file context.
        filtered_rows = []
        for row in rows:
            try:
                cites = json.loads(str(row["citations_json"] or "[]"))
            except Exception:
                cites = []
            if isinstance(cites, list) and len(cites) > 0:
                continue
            filtered_rows.append(row)

        for row in filtered_rows[-max_messages:]:
            role = str(row["role"] or "").strip().lower()
            content = str(row["content"] or "").strip()
            if not content:
                continue
            tag = "User" if role == "user" else "Assistant"
            lines.append(f"{tag}: {self._compact_context_text(content, 700)}")
        return "\n".join(lines) if lines else "(none)"

    def _accurate_langgraph_retrieval_stream(self, message: str, upload_id: str | None = None):
        if not upload_id:
            yield {"type": "detail", "line": "No upload scope found for accurate retrieval."}
            yield {
                "type": "result",
                "retrieved": [],
                "max_score": 0.0,
                "trace": {"planner": "react_graph", "queries": [], "probes": [], "rounds_used": 0, "refinements": 0},
            }
            return

        initial_queries = self._plan_queries(message)
        initial_queries = self._augment_queries_for_hidden_signals(message, initial_queries)
        initial_queries = self._augment_queries_with_file_names(message, initial_queries, upload_id)
        initial_queries = self._augment_queries_with_all_files(message, initial_queries, upload_id)
        initial_queries, used_disambiguation = self._augment_queries_for_semantic_disambiguation(message, initial_queries)
        if message not in initial_queries:
            initial_queries.insert(0, message)

        max_probes = self._dynamic_probe_budget(message, len(initial_queries))
        max_rounds = self._dynamic_react_iterations(message)
        state: dict[str, Any] = {
            "message": message,
            "upload_id": upload_id,
            "active_queries": list(dict.fromkeys(initial_queries[:max_probes])),
            "seen_queries": list(dict.fromkeys(initial_queries[:max_probes])),
            "all_sets": [],
            "probes": [],
            "merged": [],
            "max_score": 0.0,
            "rounds_used": 0,
            "refinements": 0,
            "max_rounds": max_rounds,
            "max_probes": max_probes,
            "done": False,
            "disambiguation": bool(used_disambiguation),
            "status_lines": [],
        }

        def node_plan(s: dict[str, Any]) -> dict[str, Any]:
            lines = [
                f"LangGraph ReAct started: up to {int(s.get('max_rounds', 1))} round(s), {int(s.get('max_probes', 4))} probe(s)/round."
            ]
            if s.get("disambiguation"):
                lines.append("Semantic disambiguation enabled by planner.")
            return {"status_lines": lines}

        def node_probe(s: dict[str, Any]) -> dict[str, Any]:
            current_round = int(s.get("rounds_used", 0)) + 1
            sets = list(s.get("all_sets", []))
            probes = list(s.get("probes", []))
            lines = []
            for q in list(s.get("active_queries", []))[: int(s.get("max_probes", 4))]:
                q_emb = self.embeddings.embed_one(q)
                retrieved, q_score = self.retriever.retrieve(query_embedding=q_emb, query_text=q, mode="accurate")
                sets.append((retrieved, q_score))
                probes.append(
                    {
                        "round": current_round,
                        "query": q,
                        "hits": len(retrieved),
                        "score": float(q_score or 0.0),
                    }
                )
                q_short = q if len(q) <= 90 else (q[:87] + "...")
                lines.append(f"Round {current_round} · probe: {len(retrieved)} hit(s) for \"{q_short}\".")
            merged, max_score = self._merge_retrieval_sets(sets)
            return {
                "all_sets": sets,
                "probes": probes,
                "merged": merged,
                "max_score": max_score,
                "rounds_used": current_round,
                "status_lines": lines,
            }

        def node_assess(s: dict[str, Any]) -> dict[str, Any]:
            sufficient = self._is_retrieval_sufficient(s.get("merged", []), float(s.get("max_score", 0.0)))
            rounds_used = int(s.get("rounds_used", 0))
            max_rounds_local = int(s.get("max_rounds", 1))
            done = bool(sufficient or rounds_used >= max_rounds_local)
            lines = []
            if sufficient:
                lines.append(
                    f"Coverage looks sufficient after round {rounds_used} (hits={len(s.get('merged', []))}, score={float(s.get('max_score', 0.0)):.3f})."
                )
            elif done:
                lines.append("Reached round limit; stopping retrieval iterations.")
            else:
                lines.append("Coverage still weak; refining queries for another round.")
            return {"done": done, "status_lines": lines}

        def node_refine(s: dict[str, Any]) -> dict[str, Any]:
            refined = self._refine_queries_from_retrieval(
                str(s.get("message", "")),
                s.get("merged", []),
                list(s.get("seen_queries", [])),
            )
            seen = list(s.get("seen_queries", []))
            seen_keys = {str(x).lower() for x in seen}
            next_queries: list[str] = []
            for q in refined:
                key = str(q).lower()
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                seen.append(q)
                next_queries.append(q)
            if not next_queries:
                return {
                    "done": True,
                    "seen_queries": seen,
                    "status_lines": ["No new useful probes were generated; stopping retrieval iterations."],
                }
            max_probes_local = int(s.get("max_probes", 4))
            next_queries = next_queries[:max_probes_local]
            return {
                "active_queries": next_queries,
                "seen_queries": seen,
                "refinements": int(s.get("refinements", 0)) + 1,
                "status_lines": [f"Refined query plan for next round: {len(next_queries)} probe(s)."],
            }

        graph = StateGraph(dict)
        graph.add_node("plan", RunnableLambda(node_plan))
        graph.add_node("probe", RunnableLambda(node_probe))
        graph.add_node("assess", RunnableLambda(node_assess))
        graph.add_node("refine", RunnableLambda(node_refine))
        graph.add_edge(START, "plan")
        graph.add_edge("plan", "probe")
        graph.add_edge("probe", "assess")
        graph.add_conditional_edges("assess", lambda s: "end" if bool(s.get("done")) else "refine", {"end": END, "refine": "refine"})
        graph.add_conditional_edges("refine", lambda s: "end" if bool(s.get("done")) else "probe", {"end": END, "probe": "probe"})
        compiled = graph.compile()

        current_state = dict(state)
        for update in compiled.stream(current_state, stream_mode="updates"):
            for _node, delta in update.items():
                if not isinstance(delta, dict):
                    continue
                current_state.update(delta)
                for line in delta.get("status_lines", []):
                    if str(line).strip():
                        yield {"type": "detail", "line": str(line)}

        yield {
            "type": "result",
            "retrieved": current_state.get("merged", []),
            "max_score": float(current_state.get("max_score", 0.0)),
            "trace": {
                "planner": "react_graph",
                "queries": list(current_state.get("seen_queries", [])),
                "probes": list(current_state.get("probes", [])),
                "disambiguation": bool(current_state.get("disambiguation", False)),
                "rounds_used": int(current_state.get("rounds_used", 0)),
                "refinements": int(current_state.get("refinements", 0)),
            },
        }

    def _retrieve_with_react(self, message: str, mode: str, upload_id: str | None = None):
        retrieved, max_score, _trace = self._retrieve_with_trace(message, mode, upload_id)
        return retrieved, max_score

    def _retrieve_with_trace(self, message: str, mode: str, upload_id: str | None = None):
        if mode == "fast":
            q_emb = self.embeddings.embed_one(message)
            retrieved, max_score = self.retriever.retrieve(query_embedding=q_emb, query_text=message, mode=mode)
            return retrieved, max_score, {"planner": "fast", "queries": [message]}

        # Accurate mode: generic ReAct retrieval planner (no domain-specific assumptions).
        query_plan = self._plan_queries(message)
        augment_fn = getattr(self, "_augment_queries_for_hidden_signals", None)
        if callable(augment_fn):
            query_plan = augment_fn(message, query_plan)
        query_plan = self._augment_queries_with_file_names(message, query_plan, upload_id)
        query_plan = self._augment_queries_with_all_files(message, query_plan, upload_id)
        query_plan, used_disambiguation = self._augment_queries_for_semantic_disambiguation(message, query_plan)
        if message not in query_plan:
            query_plan.insert(0, message)
        max_probes = self._dynamic_probe_budget(message, len(query_plan))
        max_rounds = self._dynamic_react_iterations(message)

        all_sets = []
        probes = []
        refinements = 0
        rounds_used = 0
        merged = []
        max_score = 0.0

        active_queries = list(query_plan[:max_probes])
        seen_queries = {q.lower() for q in active_queries}

        for round_idx in range(1, max_rounds + 1):
            rounds_used = round_idx
            round_sets = []
            for q in active_queries:
                q_emb = self.embeddings.embed_one(q)
                retrieved, q_score = self.retriever.retrieve(query_embedding=q_emb, query_text=q, mode=mode)
                round_sets.append((retrieved, q_score))
                all_sets.append((retrieved, q_score))
                probes.append(
                    {
                        "round": round_idx,
                        "query": q,
                        "hits": len(retrieved),
                        "score": float(q_score or 0.0),
                    }
                )

            merged, max_score = self._merge_retrieval_sets(all_sets)
            if self._is_retrieval_sufficient(merged, max_score):
                break

            refined = self._refine_queries_from_retrieval(message, merged, list(seen_queries))
            next_queries: list[str] = []
            for q in refined:
                key = q.lower()
                if key in seen_queries:
                    continue
                seen_queries.add(key)
                next_queries.append(q)
            if not next_queries:
                break
            refinements += 1
            active_queries = next_queries[:max_probes]

        return merged, max_score, {
            "planner": "react",
            "queries": list(seen_queries)[: max(8, max_probes)],
            "probes": probes,
            "disambiguation": used_disambiguation,
            "rounds_used": rounds_used,
            "refinements": refinements,
        }

    def _expand_followup_query(self, message: str, chat_id: str | None) -> str:
        text = (message or "").strip()
        if not text or not chat_id:
            return text

        history = self.db.list_chat_messages(chat_id)
        last_user_text = ""
        for row in reversed(history):
            if str(row["role"]) != "user":
                continue
            prev = str(row["content"] or "").strip()
            if not prev:
                continue
            if prev == text:
                continue
            last_user_text = prev
            break

        if not last_user_text:
            return text
        try:
            system = (
                "You resolve chat follow-ups for retrieval. "
                "Given previous and current user messages, decide if current depends on previous context. "
                "Only mark follow-up when the current message is NOT self-contained and requires prior context "
                "(for example pronouns like it/that/those/the former/this, or omitted subject). "
                "If current message is fully self-contained, do NOT mark follow-up. "
                "Return strict JSON with keys: is_followup (boolean), confidence (number 0..1), merged_query (string). "
                "If independent, set is_followup=false and merged_query=current."
            )
            user = (
                f"Previous user message:\n{last_user_text}\n\n"
                f"Current user message:\n{text}\n\n"
                "Decide now."
            )
            out = self.llm.chat_json(system, user, timeout=10.0)
            is_followup = bool(out.get("is_followup", False))
            confidence = float(out.get("confidence", 0.0) or 0.0)
            if is_followup and confidence >= 0.72:
                merged = str(out.get("merged_query", "")).strip()
                return merged or f"{last_user_text}\nFollow-up: {text}"
            return text
        except Exception:
            return text

    def _dynamic_probe_budget(self, message: str, planned_queries: int) -> int:
        try:
            system = (
                "Choose retrieval probe budget for a ReAct RAG step. "
                "Return strict JSON: {\"probe_budget\": integer}. "
                "Constraints: 4 <= probe_budget <= 10."
            )
            user = (
                f"Question:\n{message}\n\n"
                f"Current planned query variants: {int(planned_queries)}"
            )
            out = self.llm.chat_json(system, user, timeout=8.0)
            budget = int(out.get("probe_budget", 6))
            return max(4, min(10, budget))
        except Exception:
            token_count = len(re.findall(r"[a-z0-9_]+", (message or "").lower()))
            return 4 if token_count < 8 else (6 if token_count < 20 else 8)

    def _dynamic_react_iterations(self, message: str) -> int:
        try:
            system = (
                "Choose number of ReAct retrieval rounds. "
                "Return strict JSON: {\"rounds\": integer}. "
                "Constraints: 1 <= rounds <= 3."
            )
            user = f"Question:\n{message}\n\nPick rounds."
            out = self.llm.chat_json(system, user, timeout=8.0)
            rounds = int(out.get("rounds", 2))
            return max(1, min(3, rounds))
        except Exception:
            token_count = len(re.findall(r"[a-z0-9_]+", (message or "").lower()))
            return 2 if token_count < 14 else 3

    def _is_retrieval_sufficient(self, retrieved, max_score: float) -> bool:
        if not retrieved:
            return False
        if len(retrieved) >= 8 and max_score >= settings.low_confidence_threshold:
            return True
        doc_ids = {r.chunk.get("document_id") for r in retrieved if r.chunk.get("document_id")}
        if len(doc_ids) >= 2 and len(retrieved) >= 5 and max_score >= (settings.low_confidence_threshold * 0.85):
            return True
        return False

    def _refine_queries_from_retrieval(self, message: str, retrieved, existing_queries: list[str]) -> list[str]:
        top = retrieved[:5] if retrieved else []
        snippets = "\n".join(
            f"- {r.chunk.get('file_name','')}: {r.chunk.get('preview','')}"
            for r in top
        )
        out: list[str] = []
        seen = {q.lower().strip() for q in existing_queries}

        def add(q: str) -> None:
            candidate = (q or "").strip()
            if not candidate:
                return
            key = candidate.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(candidate)

        try:
            system = (
                "You are a retrieval strategist. Given a user question and current evidence snippets, "
                "propose 1-3 NEW search queries that explore missing angles. "
                "Keep generic; do not assume any fixed domain. "
                "Return strict JSON with key next_queries as array."
            )
            user = (
                f"Question: {message}\n"
                f"Current snippets:\n{snippets}\n"
                "Need better evidence coverage and disambiguation.\n"
            )
            data = self.llm.chat_json(system, user, timeout=18.0)
            nqs = data.get("next_queries", [])
            if isinstance(nqs, list):
                for q in nqs[:3]:
                    add(str(q))
        except Exception:
            pass

        return out[:6]

    def _augment_queries_with_file_names(self, message: str, base_queries: list[str], upload_id: str | None) -> list[str]:
        if not upload_id:
            return base_queries
        docs = self.db.get_documents_by_upload_id(upload_id)
        if not docs:
            return base_queries

        # Do not use confidence-picked filenames for query expansion.
        # Only expand with files explicitly referenced by the user message.
        explicit_docs = self._find_message_referenced_docs(message, docs)
        selected = []
        for doc in explicit_docs[:4]:
            name = str(doc["file_name"] if "file_name" in doc.keys() else "").strip()
            if not name:
                continue
            selected.append(Path(name).stem)
        if not selected:
            return base_queries

        out = [str(q).strip() for q in base_queries if str(q).strip()]
        seen = {q.lower() for q in out}
        for stem in selected:
            q = f"{message} {stem}".strip()
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out

    def _augment_queries_with_all_files(self, message: str, base_queries: list[str], upload_id: str | None) -> list[str]:
        if not upload_id:
            return base_queries
        docs = self.db.get_documents_by_upload_id(upload_id)
        if not docs:
            return base_queries

        out = [str(q).strip() for q in base_queries if str(q).strip()]
        seen = {q.lower() for q in out}

        stems: list[str] = []
        for doc in docs:
            file_name = str(doc["file_name"] if "file_name" in doc.keys() else "").strip()
            if not file_name:
                continue
            stem = Path(file_name).stem.strip()
            if stem:
                stems.append(stem)

        unique_stems = []
        seen_stems = set()
        for stem in stems:
            key = stem.lower()
            if key in seen_stems:
                continue
            seen_stems.add(key)
            unique_stems.append(stem)

        # Keep this bounded to avoid exploding probe count; deep-dig handles strict per-file search later.
        for stem in unique_stems[:12]:
            q = f"{message} in file {stem}".strip()
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(q)
        return out[:24]

    def _augment_queries_for_semantic_disambiguation(self, message: str, base_queries: list[str]) -> tuple[list[str], bool]:
        """
        Generic ambiguity expansion:
        build alternative semantic framings so retrieval doesn't overfit one label meaning.
        """
        out = [str(q).strip() for q in (base_queries or []) if str(q).strip()]
        seen = {q.lower() for q in out}

        def add(query: str) -> None:
            q = str(query or "").strip()
            if not q:
                return
            key = q.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(q)

        used_llm = False
        try:
            system = (
                "You create retrieval query variants for ambiguous user wording. "
                "Return strict JSON: {\"alt_queries\": [..]}. "
                "Rules: keep generic, no domain assumptions, and vary interpretation (taxonomy vs relation vs workflow vs configuration). "
                "Return 0 to 3 short queries."
            )
            user = (
                f"User message: {message}\n"
                f"Base queries: {json.dumps(out[:6], ensure_ascii=False)}\n"
            )
            data = self.llm.chat_json(system, user, timeout=18.0)
            alt = data.get("alt_queries", [])
            if isinstance(alt, list):
                for q in alt[:3]:
                    add(str(q))
            used_llm = True
        except Exception:
            used_llm = False

        return out[:14], used_llm or (len(out) > len(base_queries))

    def _rescue_challenge_retrieval(self, retrieved, message: str, upload_id: str | None):
        if len(retrieved) >= 2:
            return retrieved
        if not upload_id:
            return retrieved

        terms = self._extract_terms_for_rescue(message)
        if not terms:
            return retrieved

        rows = self.db.search_chunks_in_upload_by_terms(
            upload_id,
            terms,
            limit=28,
        )
        if not rows:
            return retrieved

        # Build lightweight retrieved objects with deterministic low score.
        adapter = type(retrieved[0]) if retrieved else None
        rescued = list(retrieved)
        seen_ids = {r.chunk.get("id") for r in rescued}
        for row in rows:
            chunk = dict(row)
            cid = chunk.get("id")
            if not cid or cid in seen_ids:
                continue
            seen_ids.add(cid)
            if adapter is not None:
                rescued.append(adapter(chunk=chunk, score=0.04))
            else:
                from app.retrieval import RetrievedChunk
                rescued.append(RetrievedChunk(chunk=chunk, score=0.04))
            if len(rescued) >= 14:
                break
        return rescued

    def _accurate_per_file_react_fallback(self, message: str, upload_id: str | None):
        if not upload_id:
            return []
        docs = self.db.get_documents_by_upload_id(upload_id)
        if not docs:
            return []

        query_plan = self._plan_queries(message)
        augment_fn = getattr(self, "_augment_queries_for_hidden_signals", None)
        if callable(augment_fn):
            query_plan = augment_fn(message, query_plan)
        if message not in query_plan:
            query_plan.insert(0, message)

        terms: list[str] = []
        for q in query_plan[:4]:
            terms.extend(re.findall(r"[a-z0-9_]+", q.lower()))
        terms = [t for t in list(dict.fromkeys(terms)) if len(t) >= 4][:28]
        if not terms:
            return []

        per_doc_cap = 3
        rescued: list[RetrievedChunk] = []
        seen: set[str] = set()
        for doc in docs:
            rows = self.db.search_chunks_in_document_by_terms(doc["id"], terms, limit=per_doc_cap)
            added = 0
            for row in rows:
                chunk = dict(row)
                cid = chunk.get("id")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                score = 0.03
                text_low = f"{chunk.get('text', '')} {chunk.get('preview', '')}".lower()
                overlap = sum(1 for t in terms[:14] if t in text_low)
                score += min(overlap * 0.01, 0.08)
                rescued.append(RetrievedChunk(chunk=chunk, score=score))
                added += 1
                if added >= per_doc_cap:
                    break
            if len(rescued) >= 18:
                break
        rescued.sort(key=lambda x: x.score, reverse=True)
        return rescued[:14]

    def _accurate_deep_dig_if_needed(
        self,
        message: str,
        upload_id: str | None,
        retrieved,
        max_score: float,
        mode: str = "accurate",
    ):
        if not upload_id:
            return retrieved, {"triggered": False, "files": [], "notes": []}

        docs = self.db.get_documents_by_upload_id(upload_id)
        if not docs:
            return retrieved, {"triggered": False, "files": [], "notes": []}

        current_hits = len(retrieved)
        weak_coverage = current_hits <= 2 or max_score < settings.low_confidence_threshold
        requested_docs = self._find_message_referenced_docs(message, docs)
        requested_ids = {str(d["id"]) for d in requested_docs}
        current_doc_ids = {str(r.chunk.get("document_id")) for r in retrieved if r.chunk.get("document_id")}
        missing_requested = bool(requested_ids) and not requested_ids.issubset(current_doc_ids)

        # Trigger deep dig if retrieval is weak OR user explicitly asked about files not represented yet.
        if not weak_coverage and not missing_requested:
            return retrieved, {"triggered": False, "files": [], "notes": []}

        # Accurate mode: run global deep-dig prompts across ALL files.
        # Moderate mode: let planner pick a subset.
        if mode == "accurate":
            docs_to_scan = list(docs)
        else:
            docs_to_scan = self._select_files_for_deep_dig(message, docs, retrieved, max_files=min(8, max(3, len(docs))))
            if not docs_to_scan:
                docs_to_scan = list(docs)
        notes: list[str] = []
        notes.append(
            f"Deep dig strategy: {'all files' if mode == 'accurate' else 'selected files'} "
            f"({len(docs_to_scan)} file(s))."
        )
        if requested_docs:
            req_names = [str(d["file_name"] if "file_name" in d.keys() else "unknown") for d in requested_docs[:8]]
            notes.append(f"User-mentioned files detected: {', '.join(req_names)}")

        query_plan = self._plan_queries(message)
        augment_fn = getattr(self, "_augment_queries_for_hidden_signals", None)
        if callable(augment_fn):
            query_plan = augment_fn(message, query_plan)
        query_plan = self._augment_queries_with_file_names(message, query_plan, upload_id)
        max_probes = self._dynamic_probe_budget(message, len(query_plan))
        terms: list[str] = []
        for q in query_plan[:max_probes]:
            terms.extend(re.findall(r"[a-z0-9_]+", q.lower()))
        # Add additional term candidates inferred by the model.
        terms.extend(self._extract_terms_for_rescue(message))
        terms = [t for t in list(dict.fromkeys(terms)) if len(t) >= 2][:48]
        if not terms:
            return retrieved, {"triggered": False, "files": [], "notes": notes}

        rescued = list(retrieved)
        seen = {str(r.chunk.get("id")) for r in rescued if r.chunk.get("id")}
        files_scanned: list[str] = []
        file_rank_rows: list[tuple[str, int, int]] = []

        global_file_prompts: list[str] = []
        if mode == "accurate":
            global_file_prompts = self._plan_global_file_dig_prompts(message)
            notes.append(f"Global deep-dig prompt set: {len(global_file_prompts)} prompt(s) applied to all files.")

        for doc in docs_to_scan:
            doc_id = str(doc["id"])
            doc_name = str(doc["file_name"] if "file_name" in doc.keys() else "unknown")
            files_scanned.append(doc_name)
            file_queries: list[str] = []
            if mode == "accurate":
                file_queries = list(global_file_prompts)
                notes.append(f"Dig plan for {doc_name}: global prompt set.")
            else:
                notes.append(f"Dig plan for {doc_name}: moderate scan (no per-file prompt generation).")

            full_rows = self.db.get_chunks_for_document(doc_id, limit=2000)
            if not full_rows:
                notes.append(f"Scanned {doc_name}: no chunks available.")
                continue

            page_count = len({r["anchor_page"] for r in full_rows if r["anchor_page"] is not None})
            candidate_added = 0
            per_prompt_hits: list[str] = []
            prompt_hit_sum = 0

            if mode == "accurate":
                for fq in file_queries:
                    prompt_terms = self._extract_terms_for_rescue(fq)[:48]
                    if not prompt_terms:
                        continue
                    rows = self.db.search_chunks_in_document_by_terms(doc_id, prompt_terms, limit=50)
                    prompt_hit_sum += len(rows)
                    per_prompt_hits.append(f"{(fq[:42] + '...') if len(fq) > 42 else fq}: {len(rows)} hit(s)")
                    for row in rows:
                        chunk = dict(row)
                        if str(chunk.get("anchor_type", "")) == "doc_summary":
                            continue
                        cid = str(chunk.get("id") or "")
                        if not cid or cid in seen:
                            continue
                        text_low = f"{chunk.get('text', '')} {chunk.get('preview', '')}".lower()
                        overlap = sum(1 for t in prompt_terms if t in text_low)
                        if overlap <= 0:
                            continue
                        score = 0.05 + min(overlap * 0.012, 0.22)
                        if doc_id in requested_ids:
                            score += 0.08
                        rescued.append(RetrievedChunk(chunk=chunk, score=score))
                        seen.add(cid)
                        candidate_added += 1

            # Keep current behavior too: exhaustive lexical scan as safety net.
            for row in full_rows: 
                chunk = dict(row)
                if str(chunk.get("anchor_type", "")) == "doc_summary":
                    continue
                cid = str(chunk.get("id") or "")
                if not cid or cid in seen:
                    continue
                text_low = f"{chunk.get('text', '')} {chunk.get('preview', '')}".lower()
                overlap = sum(1 for t in terms[:40] if t in text_low)
                if overlap <= 0:
                    continue
                score = 0.04 + min(overlap * 0.01, 0.16)
                if doc_id in requested_ids:
                    score += 0.08
                rescued.append(RetrievedChunk(chunk=chunk, score=score))
                seen.add(cid)
                candidate_added += 1

            if per_prompt_hits:
                notes.append(f"{doc_name} prompt probes: " + "; ".join(per_prompt_hits[:6]))
            notes.append(
                f"Scanned {doc_name}: {len(full_rows)} chunk(s), {page_count} page(s), {candidate_added} candidate hit(s)."
            )
            file_rank_rows.append((doc_name, prompt_hit_sum, candidate_added))

        if file_rank_rows:
            file_rank_rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
            top_rank = ", ".join(
                f"{name} (prompt_hits={hits}, candidates={cand})"
                for name, hits, cand in file_rank_rows[:8]
            )
            notes.append(f"File ranking by deep-dig evidence: {top_rank}")

        rescued.sort(key=lambda x: x.score, reverse=True)
        # Keep a wide candidate pool for accurate mode; moderate keeps a medium-wide pool.
        max_candidates = 80 if mode == "accurate" else 36
        return rescued[:max_candidates], {"triggered": True, "files": files_scanned, "notes": notes}

    def _select_files_for_deep_dig(self, message: str, docs: list[Any], retrieved, max_files: int = 6) -> list[Any]:
        if not docs:
            return []
        file_names = [str(d["file_name"] if "file_name" in d.keys() else "") for d in docs]
        observed = []
        for item in (retrieved or [])[:10]:
            observed.append(
                {
                    "file_name": str(item.chunk.get("file_name", "")),
                    "preview": str(item.chunk.get("preview", ""))[:220],
                }
            )
        try:
            system = (
                "Pick which files to deep-search for a user request. "
                "Return strict JSON with key file_names as an array of file names selected from available files. "
                "Prefer enough coverage; choose up to the requested limit."
            )
            user = (
                f"Question:\n{message}\n\n"
                f"Available files:\n{json.dumps(file_names, ensure_ascii=False)}\n\n"
                f"Current weak evidence:\n{json.dumps(observed, ensure_ascii=False)}\n\n"
                f"Select at most {int(max_files)} files."
            )
            out = self.llm.chat_json(system, user, timeout=20.0)
            chosen = out.get("file_names", [])
            selected: list[Any] = []
            seen = set()
            if isinstance(chosen, list):
                for name in chosen:
                    doc = self._match_doc_by_name(str(name), docs)
                    if not doc:
                        continue
                    did = str(doc["id"])
                    if did in seen:
                        continue
                    seen.add(did)
                    selected.append(doc)
                    if len(selected) >= max_files:
                        break
            if selected:
                return selected
        except Exception:
            pass
        return docs[:max_files]

    def _plan_queries_for_file_dig(self, message: str, file_name: str) -> list[str]:
        try:
            system = (
                "Generate file-specific retrieval prompts for RAG. "
                "Return strict JSON with key queries as array of 2 to 6 short prompts."
            )
            user = (
                f"User request:\n{message}\n\n"
                f"Target file:\n{file_name}\n\n"
                "Create prompts that help find exact evidence in this file."
            )
            out = self.llm.chat_json(system, user, timeout=16.0)
            queries = out.get("queries", [])
            if isinstance(queries, list):
                clean = [str(q).strip() for q in queries if str(q).strip()]
                if clean:
                    return clean[:6]
        except Exception:
            pass
        stem = Path(str(file_name or "file")).stem
        return [f"{message} in {stem}", f"exact clause for {message} {stem}", message]

    def _plan_global_file_dig_prompts(self, message: str) -> list[str]:
        try:
            system = (
                "Generate a global retrieval prompt set for deep document digging. "
                "These prompts will be run against every file. "
                "Return strict JSON with key queries as array of 4 to 10 short prompts."
            )
            user = (
                f"User request:\n{message}\n\n"
                "Generate diverse prompts that maximize recall and evidence quality across multiple files."
            )
            out = self.llm.chat_json(system, user, timeout=18.0)
            queries = out.get("queries", [])
            if isinstance(queries, list):
                clean = [str(q).strip() for q in queries if str(q).strip()]
                if clean:
                    return clean[:10]
        except Exception:
            pass
        return self._plan_queries(message)[:6] or [message]

    def _find_message_referenced_docs(self, message: str, docs: list[Any]) -> list[Any]:
        text = str(message or "").lower()
        msg_tokens = set(re.findall(r"[a-z0-9_]+", text))
        if not msg_tokens:
            return []

        scored: list[tuple[float, Any]] = []
        for doc in docs:
            file_name = str(doc["file_name"] if "file_name" in doc.keys() else "")
            stem = Path(file_name).stem.lower()
            stem_tokens = set(re.findall(r"[a-z0-9_]+", stem))
            if not stem_tokens:
                continue
            overlap = len(msg_tokens.intersection(stem_tokens))
            score = float(overlap)
            # small boost when stem appears as phrase in prompt
            if stem and stem in text:
                score += 2.5
            # support "Contract A / Contract B" style mentions
            if "contract a" in text and ("contract" in stem_tokens and "a" in stem_tokens):
                score += 3.0
            if "contract b" in text and ("contract" in stem_tokens and "b" in stem_tokens):
                score += 3.0
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _score, doc in scored[:6]]

    def _build_scope_clarification_prompt(
        self,
        message: str,
        docs: list[Any],
        requested_docs: list[Any],
        scope: dict[str, Any] | None = None,
    ) -> str | None:
        # Always auto-proceed: do not interrupt user with file-scope confirmation prompts.
        return None
        if requested_docs:
            return None
        if not docs or len(docs) < 2:
            return None
        if scope and str(scope.get("scope", "unspecified")) in {"all", "specific"}:
            return None

        if not self._needs_scope_clarification_by_llm(message, docs):
            return None

        options = [str(d["file_name"] if "file_name" in d.keys() else "unknown") for d in docs[:8]]
        options_text = "\n".join(f"- {name}" for name in options)
        return (
            "I found multiple files and your request may be ambiguous across them.\n\n"
            "Please confirm the scope before I answer:\n"
            "1. A specific file (tell me its name), or\n"
            "2. All relevant files in the current upload.\n\n"
            "Available files:\n"
            f"{options_text}"
        )

    def _infer_scope_from_context(self, message: str, chat_id: str | None, docs: list[Any]) -> dict[str, Any]:
        if not docs:
            return {"scope": "unspecified", "file_name": ""}
        file_names = [str(d["file_name"] if "file_name" in d.keys() else "") for d in docs[:16]]
        recent = []
        if chat_id:
            rows = self.db.list_chat_messages(chat_id)
            for row in rows[-8:]:
                role = str(row["role"] or "")
                content = str(row["content"] or "").strip()
                if content:
                    recent.append({"role": role, "content": content[:500]})
        try:
            system = (
                "Decide the user's intended file scope in a multi-file RAG chat. "
                "Use current message plus recent conversation and available files. "
                "Return strict JSON with keys: scope, file_name. "
                "scope must be one of: all, specific, unspecified. "
                "If scope=specific, set file_name to best matching available file name."
            )
            user = (
                f"Current message:\n{message}\n\n"
                f"Available files:\n{json.dumps(file_names, ensure_ascii=False)}\n\n"
                f"Recent conversation:\n{json.dumps(recent, ensure_ascii=False)}"
            )
            out = self.llm.chat_json(system, user, timeout=10.0)
            scope = str(out.get("scope", "unspecified")).strip().lower()
            if scope not in {"all", "specific", "unspecified"}:
                scope = "unspecified"
            file_name = str(out.get("file_name", "")).strip()
            return {"scope": scope, "file_name": file_name}
        except Exception:
            return {"scope": "unspecified", "file_name": ""}

    def _match_doc_by_name(self, file_name: str, docs: list[Any]):
        target = str(file_name or "").strip().lower()
        if not target:
            return None
        for d in docs:
            name = str(d["file_name"] if "file_name" in d.keys() else "").strip().lower()
            if name == target:
                return d
        for d in docs:
            name = str(d["file_name"] if "file_name" in d.keys() else "").strip().lower()
            if target in name or name in target:
                return d
        return None

    def _needs_scope_clarification_by_llm(self, message: str, docs: list[Any]) -> bool:
        """
        Ask the model whether the user intent is file-scope ambiguous.
        This avoids brittle, domain-specific hard-coded keyword gates.
        """
        if not message or len(docs) < 2:
            return False
        file_names = [str(d["file_name"] if "file_name" in d.keys() else "") for d in docs[:12]]
        try:
            system = (
                "You decide if a user question is ambiguous about file scope in a multi-file RAG workspace. "
                "Return strict JSON with keys: needs_clarification (boolean), reason (string). "
                "Set needs_clarification=true only when the question could reasonably target multiple files "
                "and the user did not specify one file or explicit global scope."
            )
            user = (
                f"User question:\n{message}\n\n"
                f"Available file names:\n{json.dumps(file_names, ensure_ascii=False)}\n\n"
                "Decide now."
            )
            out = self.llm.chat_json(system, user, timeout=14.0)
            return bool(out.get("needs_clarification", False))
        except Exception:
            # Conservative fallback: if many files and no explicit filename mentioned, ask clarification.
            low = str(message).lower()
            has_explicit_file_name = any(fn and fn.lower() in low for fn in file_names)
            return len(file_names) >= 3 and not has_explicit_file_name

    def _apply_explicit_file_scope(self, message: str, requested_docs: list[Any], retrieved):
        if not retrieved or not requested_docs:
            return retrieved
        low = str(message or "").lower()
        # If user explicitly asks "from/in <file>", prefer strict file scope.
        strict = any(k in low for k in (" from ", " in ", "within ", "inside "))
        if not strict:
            return retrieved
        requested_ids = {str(d["id"]) for d in requested_docs}
        scoped = [r for r in retrieved if str(r.chunk.get("document_id")) in requested_ids]
        return scoped if scoped else []

    def _exhaustive_requested_doc_scan(self, message: str, requested_docs: list[Any], limit_total: int = 24):
        if not requested_docs:
            return []
        terms = [t for t in re.findall(r"[a-z0-9_]+", (message or "").lower()) if len(t) >= 2]
        terms = list(dict.fromkeys(terms))[:64]
        out: list[RetrievedChunk] = []
        seen: set[str] = set()
        for doc in requested_docs[:4]:
            rows = self.db.get_chunks_for_document(str(doc["id"]), limit=420)
            for row in rows:
                chunk = dict(row)
                if str(chunk.get("anchor_type", "")) == "doc_summary":
                    continue
                if self._looks_like_toc(chunk):
                    continue
                cid = str(chunk.get("id") or "")
                if not cid or cid in seen:
                    continue
                seen.add(cid)
                text_low = f"{chunk.get('text', '')} {chunk.get('preview', '')}".lower()
                overlap = sum(1 for t in terms[:40] if t in text_low)
                score = 0.03 + min(overlap * 0.01, 0.2)
                if any(k in text_low for k in ("refund", "payment", "effective", "termination", "cancel", "reimburse")):
                    score += 0.04
                out.append(RetrievedChunk(chunk=chunk, score=score))
                if len(out) >= limit_total:
                    break
            if len(out) >= limit_total:
                break
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:limit_total]

    def _extract_challenges_from_citations(self, citation_map: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
        keywords = (
            "challenge", "problem", "issue", "failed", "failure", "error",
            "stuck", "blocker", "bug", "inconsisten", "cannot", "can't", "unable",
        )
        extracted: list[tuple[str, str]] = []
        seen_norm: set[str] = set()

        for cid, chunk in citation_map.items():
            text = f"{chunk.get('text', '')}\n{chunk.get('preview', '')}"
            parts = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
            for part in parts:
                s = re.sub(r"\s+", " ", str(part or "").strip())
                if len(s) < 18:
                    continue
                low = s.lower()
                if not any(k in low for k in keywords):
                    continue
                s = re.sub(r"^[\-*•\d\.\)\s]+", "", s).strip()
                norm = re.sub(r"[^a-z0-9 ]+", "", low)
                if len(norm) < 14 or norm in seen_norm:
                    continue
                seen_norm.add(norm)
                extracted.append((s, cid))
                if len(extracted) >= 20:
                    return extracted
        return extracted

    def _build_challenge_count_answer(self, extracted: list[tuple[str, str]]) -> tuple[str, list[str]]:
        count = len(extracted)
        lines = [f"I found **{count}** distinct challenge signals in the indexed evidence."]
        lines.append("")
        lines.append("Top extracted challenges:")
        for idx, (text, _cid) in enumerate(extracted[:8], start=1):
            lines.append(f"{idx}. {text}")
        answer = "\n".join(lines).strip()

        citation_ids: list[str] = []
        seen: set[str] = set()
        for _text, cid in extracted:
            if cid in seen:
                continue
            seen.add(cid)
            citation_ids.append(cid)
            if len(citation_ids) >= 6:
                break
        return answer, citation_ids

    def _build_react_notes(
        self,
        *,
        message: str,
        retrieval_message: str,
        mode: str,
        max_score: float,
        raw_hits: int,
        pre_rescue_hits: int,
        final_hits: int,
        trace: dict[str, Any],
    ) -> list[str]:
        notes: list[str] = []
        if retrieval_message != message:
            notes.append("Follow-up detected: expanded query with prior user context.")

        planner = str(trace.get("planner", "fast")).lower()
        if planner in {"react", "react_graph"}:
            probes = trace.get("probes", [])
            if bool(trace.get("disambiguation")):
                notes.append("Semantic disambiguation enabled: probing multiple interpretations of user wording.")
            rounds_used = int(trace.get("rounds_used", 1) or 1)
            refinements = int(trace.get("refinements", 0) or 0)
            notes.append(f"Iterative retrieval: {rounds_used} round(s), {refinements} refinement step(s).")
            if isinstance(probes, list) and probes:
                for idx, p in enumerate(probes, start=1):
                    q = str(p.get("query", "")).strip()
                    q = q if len(q) <= 90 else (q[:87] + "...")
                    round_no = int(p.get("round", 1) or 1)
                    notes.append(f"Round {round_no} · Probe {idx}: {int(p.get('hits', 0))} hit(s) for \"{q}\".")
        elif mode == "fast":
            notes.append("Fast mode: single-query retrieval path.")

        if raw_hits == 0:
            notes.append("No direct retrieval hit before upload-scope filtering.")
        if pre_rescue_hits == 0 and final_hits > 0:
            notes.append("Direct match was weak; switched to challenge-signal extraction from indexed chunks.")
        if pre_rescue_hits == 0 and final_hits > 0 and mode == "accurate":
            notes.append("Accurate fallback activated: probing each file individually for hidden evidence.")
        if final_hits == 0:
            notes.append("Question terms did not map to indexed evidence in current upload scope.")
        elif final_hits <= 2:
            notes.append("Evidence coverage is limited; answer may be partial.")

        if mode == "accurate" and max_score < settings.hard_not_found_threshold:
            notes.append("Similarity score is low; explicit proof is thin for this prompt.")
        return notes[:8]

    def _plan_queries(self, message: str) -> list[str]:
        system = (
            "Generate retrieval queries for RAG. Return strict JSON with key queries as an array of 1 to 3 short search queries. "
            "Queries should be semantically different and improve coverage."
        )
        user = f"Question: {message}"
        try:
            data = self.llm.chat_json(system, user, timeout=20.0)
            queries = data.get("queries", [])
            if not isinstance(queries, list):
                return [message]
            cleaned = []
            for q in queries:
                qs = str(q).strip()
                if qs:
                    cleaned.append(qs)
            return cleaned or [message]
        except Exception:
            return [message]

    def _augment_queries_for_hidden_signals(self, message: str, base_queries: list[str]) -> list[str]:
        out = [str(x).strip() for x in (base_queries or []) if str(x).strip()]
        seen = {x.lower() for x in out}

        def add(query: str) -> None:
            candidate = str(query or "").strip()
            if not candidate:
                return
            key = candidate.lower()
            if key in seen:
                return
            seen.add(key)
            out.append(candidate)

        try:
            system = (
                "Generate hidden-signal retrieval expansions for RAG. "
                "Return strict JSON with key extra_queries as array of 0..4 items. "
                "Do not assume any fixed domain."
            )
            user = (
                f"User message:\n{message}\n\n"
                f"Current query plan:\n{json.dumps(out[:8], ensure_ascii=False)}"
            )
            data = self.llm.chat_json(system, user, timeout=12.0)
            eq = data.get("extra_queries", [])
            if isinstance(eq, list):
                for q in eq[:4]:
                    add(str(q))
        except Exception:
            pass
        return out[:8]

    def _merge_retrieval_sets(self, sets):
        if not sets:
            return [], 0.0
        best_score = max((score for _, score in sets), default=0.0)
        by_chunk: dict[str, Any] = {}
        doc_counts = defaultdict(int)

        for retrieved, _ in sets:
            for item in retrieved:
                cid = item.chunk.get("id")
                if not cid:
                    continue
                current = by_chunk.get(cid)
                if current is None or item.score > current.score:
                    by_chunk[cid] = item

        ranked = sorted(by_chunk.values(), key=lambda x: x.score, reverse=True)
        final = []
        for item in ranked:
            doc_id = item.chunk.get("document_id")
            if doc_id and doc_counts[doc_id] >= 4:
                continue
            final.append(item)
            if doc_id:
                doc_counts[doc_id] += 1
            if len(final) >= 14:
                break
        return final, best_score

    def _rewrite_query(self, message: str, retrieved) -> str:
        if not retrieved:
            return message
        top = retrieved[:3]
        context = "\n".join(
            f"- {r.chunk.get('file_name','')}: {r.chunk.get('preview','')}"
            for r in top
        )
        system = (
            "Rewrite vague user questions into a concrete retrieval query for documents. "
            "Return strict JSON with key focused_query only."
        )
        user = (
            f"Question: {message}\n"
            f"Top snippets:\n{context}\n"
            "If question is already specific, keep it close. "
            "If pronouns are vague, anchor to the available document subject."
        )
        try:
            data = self.llm.chat_json(system, user, timeout=30.0)
            fq = str(data.get("focused_query", "")).strip()
            return fq or message
        except Exception:
            # Heuristic fallback for short ambiguity.
            if len(message.split()) <= 4:
                lead = top[0].chunk.get("file_name", "document")
                return f"{message} based on evidence in {lead}"
            return message

    def _citations_from_chunks(self, chunks: list[dict[str, Any]], citation_ids: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for citation_id, chunk in zip(citation_ids, chunks, strict=False):
            if str(chunk.get("anchor_type", "")) == "doc_summary":
                continue
            citation = Citation(
                citation_id=citation_id,
                chunk_id=chunk["id"],
                file_path=chunk["file_path"],
                anchor_type=chunk["anchor_type"],
                anchor_page=chunk["anchor_page"],
                anchor_section=chunk["anchor_section"],
                anchor_paragraph=chunk["anchor_paragraph"],
                anchor_row=chunk["anchor_row"],
                quoted_snippet=chunk["preview"],
            )
            out.append(citation.model_dump())
        return out

    def _compress_citation_ids(
        self,
        citation_ids: list[str],
        citation_map: dict[str, dict[str, Any]],
        mode: str,
    ) -> list[str]:
        if not citation_ids:
            return []
        max_citations = 4 if mode == "fast" else 6
        selected: list[str] = []
        seen_signatures: set[str] = set()
        for cid in citation_ids:
            chunk = citation_map.get(cid)
            if not chunk:
                continue
            preview_norm = re.sub(r"\s+", " ", str(chunk.get("preview", "")).strip().lower())[:120]
            sig = "|".join(
                [
                    str(chunk.get("file_path", "")),
                    str(chunk.get("anchor_type", "")),
                    str(chunk.get("anchor_page", "")),
                    str(chunk.get("anchor_paragraph", "")),
                    str(chunk.get("anchor_row", "")),
                    preview_norm,
                ]
            )
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            selected.append(cid)
            if len(selected) >= max_citations:
                break
        return selected

    def _fallback_citation_ids(
        self,
        citation_map: dict[str, dict[str, Any]],
        message: str,
        mode: str,
    ) -> list[str]:
        if not citation_map:
            return []
        max_citations = 4 if mode == "fast" else 6
        msg_tokens = set(re.findall(r"[a-z0-9_]+", message.lower()))

        scored: list[tuple[str, float]] = []
        for cid, chunk in citation_map.items():
            text = f"{chunk.get('text','')} {chunk.get('preview','')}".lower()
            tokens = set(re.findall(r"[a-z0-9_]+", text))
            overlap = len(msg_tokens.intersection(tokens))
            score = float(overlap)
            file_name = str(chunk.get("file_name", "")).lower()
            file_tokens = set(re.findall(r"[a-z0-9_]+", file_name))
            file_overlap = len(msg_tokens.intersection(file_tokens))
            score += float(file_overlap) * 1.2
            if chunk.get("anchor_page") is not None:
                score += 0.1  # tiny stable bias to keep deterministic ordering
            scored.append((cid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        selected: list[str] = []
        seen_sig: set[str] = set()
        for cid, _ in scored:
            ch = citation_map[cid]
            sig = "|".join(
                [
                    str(ch.get("file_path", "")),
                    str(ch.get("anchor_type", "")),
                    str(ch.get("anchor_page", "")),
                    str(ch.get("anchor_paragraph", "")),
                    str(ch.get("anchor_row", "")),
                ]
            )
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            selected.append(cid)
            if len(selected) >= max_citations:
                break
        return selected

    def _ensure_retrieval_coverage(self, message: str, retrieved):
        if not retrieved:
            return retrieved
        if not self._is_broad_question(message):
            return retrieved
        # Broad questions should use a more document-wide context, not just top header-like chunks.
        doc_ids = []
        for r in retrieved[:4]:
            did = r.chunk.get("document_id")
            if did and did not in doc_ids:
                doc_ids.append(did)
        extras = self.db.get_diverse_chunks_for_documents(doc_ids, per_doc=3, limit_total=10)
        seen = set()
        out = []
        for item in retrieved:
            cid = item.chunk.get("id")
            if cid and cid not in seen:
                seen.add(cid)
                out.append(item)
        for row in extras:
            chunk = dict(row)
            cid = chunk.get("id")
            if cid in seen:
                continue
            seen.add(cid)
            out.append(type(retrieved[0])(chunk=chunk, score=0.05))
        return out

    def _ensure_citation_coverage(
        self,
        citation_ids: list[str],
        citation_map: dict[str, dict[str, Any]],
        message: str,
        mode: str,
    ) -> list[str]:
        if not self._is_broad_question(message):
            return citation_ids
        # Broad queries should use multiple citations if available.
        if len(citation_ids) >= 2:
            return citation_ids

        max_citations = 4 if mode == "fast" else 6
        for cid in citation_map.keys():
            if cid not in citation_ids:
                citation_ids.append(cid)
            if len(citation_ids) >= max_citations:
                break
        return citation_ids[:max_citations]

    def _is_broad_question(self, message: str) -> bool:
        tokens = re.findall(r"[a-z0-9_]+", (message or "").lower())
        if len(tokens) <= 4:
            return True
        try:
            system = (
                "Decide if a user question is broad/open-ended vs narrow/factoid for retrieval coverage. "
                "Return strict JSON: {\"is_broad\": boolean}."
            )
            user = f"Question:\n{message}"
            out = self.llm.chat_json(system, user, timeout=8.0)
            return bool(out.get("is_broad", False))
        except Exception:
            return False

    def _extract_terms_for_rescue(self, message: str) -> list[str]:
        try:
            system = (
                "Extract compact lexical search terms from a question for fallback retrieval. "
                "Return strict JSON: {\"terms\": [..]}. "
                "Use 4 to 14 terms."
            )
            user = f"Question:\n{message}"
            out = self.llm.chat_json(system, user, timeout=10.0)
            terms = out.get("terms", [])
            if isinstance(terms, list):
                clean = []
                seen = set()
                for t in terms:
                    s = re.sub(r"[^a-z0-9_\\- ]+", " ", str(t).lower()).strip()
                    if not s:
                        continue
                    for part in s.split():
                        if len(part) < 2:
                            continue
                        if part in seen:
                            continue
                        seen.add(part)
                        clean.append(part)
                if clean:
                    return clean[:24]
        except Exception:
            pass
        tokens = re.findall(r"[a-z0-9_]+", (message or "").lower())
        uniq = []
        seen = set()
        for t in tokens:
            if len(t) < 3:
                continue
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq[:16]

    @staticmethod
    def _filter_by_upload_scope(retrieved, upload_id: str | None):
        # Safety: if no active upload scope is provided, do not leak old indexed folders.
        if not upload_id:
            return []
        scoped = []
        for r in retrieved:
            fp = str(r.chunk.get("file_path", ""))
            if upload_id in fp:
                scoped.append(r)
        # Strict scope: if nothing matches current upload, return empty.
        return scoped

    @staticmethod
    def _strip_doc_summary_chunks(retrieved):
        if not retrieved:
            return []
        out = []
        for item in retrieved:
            if str(item.chunk.get("anchor_type", "")) == "doc_summary":
                continue
            out.append(item)
        return out

    def _strip_toc_chunks(self, retrieved):
        if not retrieved:
            return []
        out = []
        for item in retrieved:
            if self._looks_like_toc(item.chunk):
                continue
            out.append(item)
        return out

    @staticmethod
    def _looks_like_toc(chunk: dict[str, Any]) -> bool:
        text = f"{chunk.get('text', '')}\n{chunk.get('preview', '')}"
        low = re.sub(r"\s+", " ", text.strip().lower())
        if not low:
            return False

        if (
            "table of contents" in low
            or low.startswith("contents")
            or low.startswith("acknowledgment")
            or low.startswith("acknowledgement")
            or low.startswith("preface")
            or low.startswith("foreword")
            or low.startswith("list of figures")
            or low.startswith("list of tables")
            or low.startswith("copyright")
        ):
            return True

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullet_lines = [ln for ln in lines if ln.startswith(("•", "-", "*"))]
        if len(bullet_lines) >= 6:
            signal_terms = ("typo", "suggest", "correction", "chapter", "thanks", "thank")
            if sum(1 for t in signal_terms if t in low) >= 2:
                return True

        if len(lines) < 3:
            return False

        toc_like_lines = 0
        for ln in lines[:20]:
            has_dot_leader = ("..." in ln) or (" . " in ln)
            ends_with_page = bool(re.search(r"\b\d{1,4}\s*$", ln))
            starts_with_index = bool(re.match(r"^\s*(chapter|part|appendix|\d+(\.\d+){0,3})\b", ln.lower()))
            if ends_with_page and (has_dot_leader or starts_with_index):
                toc_like_lines += 1
        return toc_like_lines >= 3

    @staticmethod
    def _chunk_location(chunk: dict[str, Any]) -> str:
        if chunk.get("anchor_type") == "doc_summary":
            return "document summary"
        if chunk.get("anchor_type") == "pdf_page":
            return f"page {chunk.get('anchor_page')}, paragraph {chunk.get('anchor_paragraph')}"
        if chunk.get("anchor_type") in {"image", "image_source", "pdf_image", "pdf_image_source"}:
            return f"image source page {chunk.get('anchor_page')}, segment {chunk.get('anchor_paragraph')}"
        if chunk.get("anchor_type") == "ppt_slide":
            return f"slide {chunk.get('anchor_page')}, block {chunk.get('anchor_paragraph')}"
        if chunk.get("anchor_type") in {"md_heading", "docx_paragraph", "txt_block"}:
            sec = chunk.get("anchor_section")
            para = chunk.get("anchor_paragraph")
            if sec:
                return f"section {sec}, paragraph {para}"
            return f"paragraph {para}"
        return "unknown"
