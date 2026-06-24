from pathlib import Path
from app.parser import DocumentParser

pdf_path = Path("sample_test.pdf")
if not pdf_path.exists():
    print("Error: sample_test.pdf not found in root directory.")
else:
    parser = DocumentParser()
    text, pages = parser.load_pdf(pdf_path)
    print(f"Total Pages Extracted: {len(pages)}")
    
    chunks = parser.create_chunks(text, source_name=pdf_path.name, pages=pages)
    print(f"Total Chunks Created: {len(chunks)}")
    print("Sample Chunk 1 Content:\n", chunks[0].page_content[:200])