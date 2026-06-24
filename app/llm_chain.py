import time
import random
import logging
from typing import Generator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from app.config import settings
from app.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Basic keywords to filter out greeting and casual inputs
CASUAL_TRIGGERS = {
    "hi", "hii", "hiii", "hello", "hey", "yo", "sup", "heya",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r u", "how are u", "what's up", "whats up",
    "who are you", "what are you", "tell me about yourself",
    "thanks", "thank you", "thx", "ty", "bye", "goodbye",
    "ok", "okay", "great", "cool", "nice", "awesome", "sounds good"
}

QUESTION_SIGNALS = {
    "?", "what", "how", "why", "when", "where", "who",
    "explain", "show", "list", "tell", "describe", "summarize"
}

def check_casual_query(query: str) -> bool:
    clean_q = query.strip().lower().rstrip("!?.")
    if clean_q in CASUAL_TRIGGERS:
        return True
    if len(clean_q.split()) <= 4 and not any(sig in clean_q for sig in QUESTION_SIGNALS):
        return True
    return False

# Prompt templates
QA_SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer questions using ONLY the retrieved document chunks provided below.

RULES:
1. Use only facts explicitly stated in the retrieved chunks.
2. Cite every factual claim: (Doc N, Source: filename.pdf, Page: N)
3. Partial answers are allowed — answer what IS present, note what is missing.
4. If the retrieved chunks do not contain the answer:
   Return exactly: "This information is not present in the provided documents."
5. Never use training knowledge. Never guess. Never infer beyond what is written.

Retrieved chunks:
{context}"""

QA_PROMPT_CASUAL = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Respond naturally and warmly. Keep it brief."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])


class LLMChainManager:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=settings.GROQ_API_KEY,
            streaming=True,
            temperature=0.0,
            max_tokens=2048,
        )
        self.sessions: dict[str, list] = {}

    def _get_messages(self, session_id: str) -> list:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def _save_turn(self, session_id: str, human: str, ai: str):
        history = self._get_messages(session_id)
        history.append(HumanMessage(content=human))
        history.append(AIMessage(content=ai))
        if len(history) > 10:
            self.sessions[session_id] = history[-10:]

    def clear_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    def _retrieve_with_retry(
        self,
        retriever: HybridRetriever,
        query: str,
        top_k: int,
        max_retries: int = 3,
    ) -> list[tuple[Document, float]]:
        delay = 1.0
        last_error = None
        for attempt in range(max_retries):
            try:
                return retriever.get_relevant_documents(query, top_k=top_k)
            except Exception as e:
                # Handle rate limits (429 / quota issues)
                if any(x in str(e).lower() for x in ("429", "resource_exhausted", "quota")):
                    if attempt < max_retries - 1:
                        time.sleep(delay + random.uniform(0, 0.5))
                        delay *= 2
                        last_error = e
                        continue
                raise
        raise last_error

    def _build_context(self, docs_with_scores: list[tuple[Document, float]]) -> str:
        blocks = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            content = doc.page_content
            
            # Smart truncation at word boundary if chunk exceeds limits
            if len(content) > settings.MAX_CHUNK_CHARS:
                content = content[: settings.MAX_CHUNK_CHARS].rsplit(" ", 1)[0]
                
            blocks.append(
                f"[DOCUMENT {i} | Source: {source} | Page: {page} | Score: {score:.2f}]\n"
                f"{content}\n"
                f"[END DOCUMENT {i}]"
            )
        return "\n\n".join(blocks)

    def get_response_stream(
        self,
        query: str,
        session_id: str,
        retriever: HybridRetriever,
        top_k: int = None,
    ) -> Generator[str, None, None]:

        if top_k is None:
            top_k = settings.TOP_K

        history = self._get_messages(session_id)

        # Bypass retrieval for general greetings or small talk
        if check_casual_query(query):
            chain = QA_PROMPT_CASUAL | self.llm | StrOutputParser()
            tokens = []
            for token in chain.stream({"input": query, "chat_history": history}):
                tokens.append(token)
                yield token
            self._save_turn(session_id, human=query, ai="".join(tokens))
            return

        # RAG workflow
        try:
            docs_with_scores = self._retrieve_with_retry(retriever, query, top_k)
        except Exception as e:
            yield f"Retrieval error: {e}"
            return

        if not docs_with_scores:
            yield "This information is not present in the provided documents."
            return

        context = self._build_context(docs_with_scores)
        tokens = []

        chain = QA_PROMPT | self.llm | StrOutputParser()
        for token in chain.stream({
            "input": query,
            "context": context,
            "chat_history": history,
        }):
            tokens.append(token)
            yield token

        self._save_turn(session_id, human=query, ai="".join(tokens))