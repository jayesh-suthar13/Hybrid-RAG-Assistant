import time
import uuid
import tempfile
from pathlib import Path
import streamlit as st

from app.config import settings
from app.parser import DocumentParser
from app.database import VectorStoreManager
from app.retriever import HybridRetriever
from app.llm_chain import LLMChainManager

st.set_page_config(page_title="Hybrid-RAG-Assistant", layout="wide")

st.markdown("""
<style>
section[data-testid="stSidebar"] { background: #111317; }
.header-box { background: #1a1f2e; padding: 1.2rem 1.5rem; border-radius: 8px; margin-bottom: 1rem; }
.header-box h1 { margin: 0; font-size: 1.5rem; }
.kb-status { padding: 4px 10px; border-radius: 6px; font-size: 0.8rem; }
.kb-ready { background: #1b3a1b; color: #7ee787; }
.kb-empty { background: #2b2b2b; color: #999; }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "db_manager" not in st.session_state:
        with st.spinner("Setting up vector store..."):
            st.session_state.db_manager = VectorStoreManager()
    if "retriever" not in st.session_state:
        with st.spinner("Loading reranker model..."):
            st.session_state.retriever = HybridRetriever(st.session_state.db_manager)
    if "llm_manager" not in st.session_state:
        st.session_state.llm_manager = LLMChainManager()
    if "uploaded_names" not in st.session_state:
        st.session_state.uploaded_names = []
    if "doc_count" not in st.session_state:
        try:
            st.session_state.doc_count = (
                st.session_state.db_manager.get_vector_store()._collection.count()
            )
        except Exception:
            st.session_state.doc_count = 0


def handle_upload(file):
    if file.name in st.session_state.uploaded_names:
        st.warning(f"{file.name} is already in the knowledge base.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getbuffer())
        tmp_path = Path(tmp.name)

    with st.spinner(f"Processing {file.name}..."):
        try:
            parser = DocumentParser()
            full_text, pages = parser.load_pdf(tmp_path)
            chunks = parser.create_chunks(
                full_text, source_name=file.name, pages=pages
            )
            st.session_state.db_manager.add_documents(chunks)
            st.session_state.uploaded_names.append(file.name)
            st.session_state.doc_count = (
                st.session_state.db_manager.get_vector_store()._collection.count()
            )
            tmp_path.unlink(missing_ok=True)
            st.success(f"Added {len(chunks)} chunks from {file.name}")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            tmp_path.unlink(missing_ok=True)
            st.error(f"Failed to process file: {e}")


def sidebar():
    with st.sidebar:
        st.markdown("### Hybrid-RAG-Assistant")
        st.caption("Dense + sparse retrieval · cross-encoder reranking")
        st.divider()

        count = st.session_state.get("doc_count", 0)
        if count:
            st.markdown(
                f'<span class="kb-status kb-ready">{count} chunks indexed</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="kb-status kb-empty">No documents yet</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.uploaded_names:
            with st.expander("Uploaded files"):
                for name in st.session_state.uploaded_names:
                    st.write(f"📄 {name}")

        st.divider()
        st.markdown("**Upload a PDF**")
        file = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")
        if file and st.button("Process document", use_container_width=True):
            handle_upload(file)

        st.divider()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.llm_manager.clear_session(st.session_state.session_id)
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()


def chat_area():
    st.markdown(
        '<div class="header-box"><h1>Hybrid-RAG-Assistant</h1>'
        '<p style="margin:4px 0 0;color:#888;font-size:0.85rem;">'
        "Ask questions about your documents, or just say hi.</p></div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.get("doc_count", 0):
        st.info("No documents loaded yet — upload a PDF from the sidebar. You can still chat!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Ask something...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            collected = ""
            try:
                stream = st.session_state.llm_manager.get_response_stream(
                    query=query,
                    session_id=st.session_state.session_id,
                    retriever=st.session_state.retriever,
                    top_k=settings.TOP_K,
                )
                for token in stream:
                    collected += token
                    placeholder.markdown(collected + "▌")
                placeholder.markdown(collected)
            except Exception as e:
                collected = f"Something went wrong: {e}"
                placeholder.error(collected)

        st.session_state.messages.append({"role": "assistant", "content": collected})


def main():
    if not getattr(settings, "GROQ_API_KEY", None):
        st.error("GROQ_API_KEY is missing — add it to your .env file.")
        st.stop()

    init_state()
    sidebar()
    chat_area()


if __name__ == "__main__":
    main()