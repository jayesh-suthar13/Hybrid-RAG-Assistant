import hashlib
import logging
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from app.database import VectorStoreManager
from app.config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

def log_debug_info(title: str, items: list[tuple[Document, float]], n: int = 10):
    if not settings.DEBUG_RETRIEVAL:
        return
    logger.info(f"\n{'='*50}\n--- {title} ---")
    if not items:
        logger.info("  No items found.")
        return
    for i, (doc, score) in enumerate(items[:n]):
        src = doc.metadata.get("source", "?")
        pg = doc.metadata.get("page", "?")
        preview = doc.page_content[:120].replace("\n", " ")
        logger.info(f"  {i+1}. [score={score:.5f}] (p.{pg} | {src}) -> {preview}...")
    logger.info('='*50)


class HybridRetriever:
    def __init__(self, db_manager: VectorStoreManager):
        self.db_manager = db_manager
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    @staticmethod
    def _get_chunk_hash(doc: Document) -> str:
        # Avoid duplicate chunks by hashing content + source metadata
        payload = (doc.metadata.get("source", "") + doc.page_content).encode()
        return hashlib.md5(payload).hexdigest()

    def _dense_retrieve(self, query: str, fetch_k: int) -> list[tuple[Document, float]]:
        try:
            results = self.db_manager.get_vector_store().similarity_search_with_score(query, k=fetch_k)
            log_debug_info("DENSE RETRIEVAL", results)
            return results
        except Exception as e:
            logger.warning(f"Dense retrieval exception: {e}")
            return []

    def _sparse_retrieve(self, query: str, fetch_k: int) -> list[tuple[Document, float]]:
        bm25 = self.db_manager.get_bm25_index()
        all_docs = self.db_manager.get_all_docs()
        if not bm25 or not all_docs:
            logger.warning("BM25 index not initialized or empty.")
            return []

        scores = bm25.get_scores(query.lower().split())
        top_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:fetch_k]
        
        results = [
            (all_docs[i], float(s))
            for i, s in top_indices
            if s > 0 and i < len(all_docs)
        ]
        log_debug_info("BM25 RETRIEVAL", results)
        return results

    def _apply_rrf(
        self,
        dense_results: list[tuple[Document, float]],
        sparse_results: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        rrf_scores = {}
        doc_map = {}

        for results in (dense_results, sparse_results):
            for rank, (doc, _) in enumerate(results):
                key = self._get_chunk_hash(doc)
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rank + RRF_K)
                doc_map[key] = doc

        fused = sorted(
            [(doc_map[k], score) for k, score in rrf_scores.items()],
            key=lambda x: x[1], reverse=True,
        )
        log_debug_info("RRF FUSED STAGE", fused)
        return fused

    def get_relevant_documents(self, query: str, top_k: int = 6) -> list[tuple[Document, float]]:
        if not query.strip():
            return []

        fetch_k = top_k * settings.FETCH_K_MULTIPLIER

        fused_docs = self._apply_rrf(
            self._dense_retrieve(query, fetch_k),
            self._sparse_retrieve(query, fetch_k),
        )
        if not fused_docs:
            return []

        # Rerank items using CrossEncoder
        pairs = [(query, doc.page_content) for doc, _ in fused_docs]
        ce_scores = self.cross_encoder.predict(pairs).tolist()

        ranked = sorted(
            zip([d for d, _ in fused_docs], ce_scores),
            key=lambda x: x[1], reverse=True,
        )
        log_debug_info("POST-RERANK STAGE", list(ranked))

        # Fallback safeguard: Keep top_k entries if all fall under threshold
        filtered = [(d, s) for d, s in ranked if s > settings.RERANK_SCORE_THRESHOLD]
        result = (filtered if filtered else list(ranked))[:top_k]

        if settings.DEBUG_RETRIEVAL:
            logger.info(f"\n--- FINAL RETRIEVAL CONTEXT ({len(result)} chunks) ---")
            total_chars = 0
            for i, (doc, score) in enumerate(result):
                chars = min(len(doc.page_content), settings.MAX_CHUNK_CHARS)
                total_chars += chars
                pg = doc.metadata.get("page", "?")
                logger.info(f"  [{i+1}] [CE Score: {score:.4f}] Page {pg} | Chars: {chars}")
            logger.info(f"  Total pipeline context chars: {total_chars}\n")

        return result