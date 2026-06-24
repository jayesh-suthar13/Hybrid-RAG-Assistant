import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    VECTOR_DB_DIR: str = os.path.join(BASE_DIR, "chroma_db")
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 6
    FETCH_K_MULTIPLIER: int = 8     
    RERANK_SCORE_THRESHOLD: float = -9.0
    MAX_CHUNK_CHARS: int = 2000
    DEBUG_RETRIEVAL: bool = False

    def validate(self):
        if not self.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in your .env file.")


settings = Settings()