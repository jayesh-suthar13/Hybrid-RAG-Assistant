from pathlib import Path
from app.parser import DocumentParser
from app.database import VectorStoreManager

pdf_path = Path("sample_test.pdf")
db = VectorStoreManager()
parser = DocumentParser()

text, pages = parser.load_pdf(pdf_path)
chunks = parser.create_chunks(text, source_name=pdf_path.name, pages=pages)

print("Indexing documents...")
db.add_documents(chunks)

# Verify counts
count = db.get_vector_store()._collection.count()
print(f"Chroma DB Vector Count: {count}")
print("BM25 Index Status:", "Ready" if db.get_bm25_index() else "Failed")