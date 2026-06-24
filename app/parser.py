from pathlib import Path
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


class DocumentParser:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_pdf(self, file_path: str | Path) -> tuple[str, list[dict]]:
        target = Path(file_path)
        reader = PdfReader(str(target))
        pages = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"text": text.strip(), "page": page_num})
        if not pages:
            raise ValueError(f"No extractable text found in '{target.name}'")
        full_text = "\n\n".join(p["text"] for p in pages)
        return full_text, pages

    def create_chunks(
        self,
        text: str,
        source_name: str = "unknown",
        pages: list[dict] | None = None,
    ) -> list[Document]:
        # Build page boundary map for metadata tagging
        page_boundaries: list[tuple[int, int, int]] = []
        if pages:
            pos = 0
            for p in pages:
                start = pos
                end = pos + len(p["text"])
                page_boundaries.append((start, end, p["page"]))
                pos = end + 2  # +2 for "\n\n" join

        raw_chunks = self.splitter.create_documents(
            texts=[text],
            metadatas=[{"source": source_name}],
        )

        # Attach page number to each chunk
        char_pos = 0
        for chunk in raw_chunks:
            page_num = 1
            if page_boundaries:
                for start, end, pnum in page_boundaries:
                    if start <= char_pos <= end:
                        page_num = pnum
                        break
            chunk.metadata["page"] = page_num
            char_pos += len(chunk.page_content)

        return raw_chunks

    def process_document(self, file_path: str | Path) -> list[Document]:
        target = Path(file_path)
        full_text, pages = self.load_pdf(target)
        return self.create_chunks(full_text, source_name=target.name, pages=pages)