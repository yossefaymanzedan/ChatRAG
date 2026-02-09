from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.database import Database
from app.vector_store import VectorStore


MODE_CONFIG = {
    "fast": {"top_k_vec": 10, "top_k_fts": 10, "final_k": 8, "use_mmr": False},
    "accurate": {"top_k_vec": 15, "top_k_fts": 15, "final_k": 12, "use_mmr": True},
}


@dataclass
class RetrievedChunk:
    chunk: dict[str, Any]
    score: float


class HybridRetriever:
    def __init__(self, db: Database, vector_store: VectorStore) -> None:
        self.db = db
        self.vector_store = vector_store

    def retrieve(self, query_embedding: list[float], query_text: str, mode: str) -> tuple[list[RetrievedChunk], float]:
        cfg = MODE_CONFIG.get(mode, MODE_CONFIG["fast"])

        vec_res = self.vector_store.query(query_embedding=query_embedding, n_results=cfg["top_k_vec"])
        vec_ids = (vec_res.get("ids") or [[]])[0]
        vec_distances = (vec_res.get("distances") or [[]])[0]
        vec_embeddings = (vec_res.get("embeddings") or [[]])[0]

        vec_scores = {
            cid: max(0.0, 1.0 - float(dist))
            for cid, dist in zip(vec_ids, vec_distances, strict=False)
        }

        fts_rows = self.db.lexical_search(query_text, cfg["top_k_fts"])
        fts_scores_raw = {row["chunk_id"]: float(row["bm25_score"]) for row in fts_rows}
        fts_scores = self._normalize_bm25(fts_scores_raw)

        all_ids = list(dict.fromkeys([*vec_scores.keys(), *fts_scores.keys()]))
        combined: dict[str, float] = {}
        for cid in all_ids:
            combined[cid] = (0.6 * vec_scores.get(cid, 0.0)) + (0.4 * fts_scores.get(cid, 0.0))

        if cfg["use_mmr"]:
            chunk_vec_map = {cid: emb for cid, emb in zip(vec_ids, vec_embeddings, strict=False) if emb is not None}
            ranked_ids = self._mmr_rank(
                query_embedding=np.array(query_embedding),
                candidates=combined,
                embeddings=chunk_vec_map,
                final_k=cfg["final_k"],
            )
        else:
            ranked_ids = [cid for cid, _ in sorted(combined.items(), key=lambda x: x[1], reverse=True)[: cfg["final_k"]]]

        chunks = self.db.get_chunks_by_ids(ranked_ids)
        score_map = combined

        results = [RetrievedChunk(chunk=dict(row), score=score_map.get(row["id"], 0.0)) for row in chunks]
        max_score = max(score_map.values()) if score_map else 0.0
        return results, max_score

    @staticmethod
    def _normalize_bm25(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        min_v = min(vals)
        max_v = max(vals)
        if max_v == min_v:
            return {k: 1.0 for k in scores}
        return {k: 1.0 - ((v - min_v) / (max_v - min_v)) for k, v in scores.items()}

    def _mmr_rank(
        self,
        query_embedding: np.ndarray,
        candidates: dict[str, float],
        embeddings: dict[str, list[float]],
        final_k: int,
        lambda_mult: float = 0.7,
    ) -> list[str]:
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        pool_ids = [cid for cid, _ in ranked]
        if not pool_ids:
            return []

        selected: list[str] = []
        remaining = pool_ids[:]

        doc_counts: dict[str, int] = {}
        rows = self.db.get_chunks_by_ids(pool_ids)
        doc_map = {r["id"]: r["document_id"] for r in rows}

        while remaining and len(selected) < final_k:
            if not selected:
                best = remaining.pop(0)
                selected.append(best)
                doc_id = doc_map.get(best)
                if doc_id:
                    doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
                continue

            best_id = None
            best_score = -1e9
            for cid in list(remaining):
                doc_id = doc_map.get(cid)
                if doc_id and doc_counts.get(doc_id, 0) >= 4:
                    continue

                rel = candidates.get(cid, 0.0)
                if cid in embeddings:
                    cand = np.array(embeddings[cid])
                    diversity = max(
                        self._cosine(cand, np.array(embeddings[sid]))
                        for sid in selected
                        if sid in embeddings
                    ) if any(sid in embeddings for sid in selected) else 0.0
                else:
                    diversity = 0.0
                mmr = (lambda_mult * rel) - ((1.0 - lambda_mult) * diversity)
                if mmr > best_score:
                    best_score = mmr
                    best_id = cid

            if best_id is None:
                best_id = remaining[0]

            remaining.remove(best_id)
            selected.append(best_id)
            doc_id = doc_map.get(best_id)
            if doc_id:
                doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        return selected

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
