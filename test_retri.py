from app.database import VectorStoreManager
from app.retriever import HybridRetriever

db = VectorStoreManager()
retriever = HybridRetriever(db)

query = "What did the candidate do at Haroba Interior?"
print(f"Searching for: '{query}'")

results = retriever.get_relevant_documents(query, top_k=3)
for idx, (doc, score) in enumerate(results, 1):
    print(f"\n[{idx}] Score: {score:.4f} | Page: {doc.metadata.get('page')}")
    print(f"Content: {doc.page_content[:120]}...")