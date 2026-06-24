import os

try:
    from dotenv import load_dotenv
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(BASE_DIR, ".env"))
except ImportError:
    pass  # Streamlit Cloud pe dotenv nahi hoga, koi baat nahi

def _get_groq_key():
    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

class Settings:
    GROQ_API_KEY: str = _get_groq_key()
    VECTOR_DB_DIR: str = "/tmp/chroma_db"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 6
    FETCH_K_MULTIPLIER: int = 8
    RERANK_SCORE_THRESHOLD: float = -9.0
    MAX_CHUNK_CHARS: int = 2000
    DEBUG_RETRIEVAL: bool = False

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing. Add it in Streamlit secrets.")

settings = Settings()