import pickle
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from app.config import settings

COLLECTION_NAME = "hybrid_rag_storage"
BM25_INDEX_FILE = Path(settings.VECTOR_DB_DIR) / "bm25_index.pkl"
BM25_CORPUS_FILE = Path(settings.VECTOR_DB_DIR) / "bm25_corpus.pkl"
BM25_DOCS_FILE = Path(settings.VECTOR_DB_DIR) / "bm25_docs.pkl"


class VectorStoreManager:
    def __init__(self):
        self.vector_db_dir = Path(settings.VECTOR_DB_DIR)
        chroma_dir = self.vector_db_dir / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(chroma_dir),
        )

        self._load_bm25_state()

    def _load_bm25_state(self):
        if (
            BM25_INDEX_FILE.exists()
            and BM25_CORPUS_FILE.exists()
            and BM25_DOCS_FILE.exists()
        ):
            try:
                with open(BM25_INDEX_FILE, "rb") as f:
                    self.bm25_index = pickle.load(f)
                with open(BM25_CORPUS_FILE, "rb") as f:
                    self.bm25_corpus = pickle.load(f)
                with open(BM25_DOCS_FILE, "rb") as f:
                    self.all_docs: list[Document] = pickle.load(f)
                return
            except Exception:
                pass

        self.bm25_index = None
        self.bm25_corpus: list[list[str]] = []
        self.all_docs: list[Document] = []

    def _save_bm25_state(self):
        with open(BM25_INDEX_FILE, "wb") as f:
            pickle.dump(self.bm25_index, f)
        with open(BM25_CORPUS_FILE, "wb") as f:
            pickle.dump(self.bm25_corpus, f)
        with open(BM25_DOCS_FILE, "wb") as f:
            pickle.dump(self.all_docs, f)

    def add_documents(self, documents: list[Document]):
        if not documents:
            return

        # No batching or sleep needed — local model, zero API calls
        self.vector_store.add_documents(documents)

        for doc in documents:
            tokens = doc.page_content.lower().split()
            if tokens:
                self.bm25_corpus.append(tokens)
                self.all_docs.append(doc)

        self.bm25_index = BM25Okapi(self.bm25_corpus)
        self._save_bm25_state()

    def get_vector_store(self) -> Chroma:
        return self.vector_store

    def get_bm25_index(self) -> BM25Okapi | None:
        return self.bm25_index

    def get_all_docs(self) -> list[Document]:
        return self.all_docs