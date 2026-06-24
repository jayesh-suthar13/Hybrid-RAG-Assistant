import uuid
from app.database import VectorStoreManager
from app.retriever import HybridRetriever
from app.llm_chain import LLMChainManager

db = VectorStoreManager()
retriever = HybridRetriever(db)
llm_chain = LLMChainManager()
session_id = str(uuid.uuid4())

queries = ["Hi there!", "Summarize the document key points."]

for q in queries:
    print(f"\nUser: {q}")
    print("Bot: ", end="")
    stream = llm_chain.get_response_stream(q, session_id, retriever)
    for token in stream:
        print(token, end="", flush=True)
    print()