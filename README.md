# 🚀 Hybrid RAG Assistant

A Hybrid Retrieval-Augmented Generation (RAG) Assistant that combines semantic search, keyword-based retrieval, and cross-encoder re-ranking to provide accurate, context-aware answers from PDF documents.

The system is designed to improve retrieval quality by leveraging both dense and sparse search techniques before passing the most relevant document chunks to a Large Language Model (LLM) for response generation.

---

## ✨ Key Features

### 🔍 Hybrid Retrieval

* Dense semantic retrieval using HuggingFace embeddings and ChromaDB
* Sparse keyword retrieval using BM25
* Reciprocal Rank Fusion (RRF) for combining retrieval results

### 🎯 Cross-Encoder Re-ranking

* Re-ranks retrieved chunks based on query-document relevance
* Improves retrieval precision
* Filters less relevant context before answer generation

### 📄 Document Intelligence

* PDF document upload and processing
* Automated text extraction and chunking
* Efficient indexing for fast retrieval

### 🤖 Context-Grounded Question Answering

* Retrieval-Augmented Generation (RAG)
* Answers generated only from retrieved document context
* Reduced hallucinations through retrieval validation

### ⚡ Optimized Query Handling

* Intent classification for greetings and casual conversations
* Bypasses retrieval for non-document queries
* Improves response speed and reduces unnecessary computation

### 🛡️ Reliability Features

* Structured prompt engineering
* Graceful handling of unavailable information
* Retry mechanism for API rate-limit scenarios

---

## 🏗️ System Architecture

```text
User Query
     │
     ▼
Intent Classification
     │
     ├── Casual Query → Direct Response
     │
     └── Document Query
             │
             ▼
      Hybrid Retrieval
      ├── Dense Search
      └── BM25 Search
             │
             ▼
    Reciprocal Rank Fusion
             │
             ▼
 Cross-Encoder Re-ranking
             │
             ▼
     Top Relevant Chunks
             │
             ▼
            LLM
             │
             ▼
       Final Answer
```

---

## 📂 Project Structure

```text
Hybrid-RAG-Assistant/
│
├── app/
│   ├── config.py
│   ├── database.py
│   ├── parser.py
│   ├── retriever.py
│   ├── llm_chain.py
│   └── ui.py
│
├── main.py
├── run.py
├── pyproject.toml
├── .env
└── README.md
```

---

## 🛠️ Tech Stack

| Category        | Technologies                      |
| --------------- | --------------------------------- |
| Programming     | Python                            |
| Framework       | LangChain                         |
| Vector Database | ChromaDB                          |
| Embeddings      | HuggingFace Sentence Transformers |
| Retrieval       | Rank-BM25, RRF                    |
| Re-ranking      | Cross-Encoder (MS MARCO MiniLM)   |
| LLM             | Groq (Llama 3)                    |
| UI              | Streamlit                         |

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jayesh-suthar13/Hybrid-RAG-Assistant.git
cd Hybrid-RAG-Assistant
```

### 2. Create Virtual Environment

```bash
uv venv
```

### 3. Install Dependencies

```bash
uv pip install -e .
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run app/ui.py
```

The application will launch locally and allow users to upload PDF documents and interact with them through a chat-based interface.

---

## 🔄 Retrieval Workflow

1. User uploads one or more PDF documents.
2. Documents are parsed, chunked, and indexed.
3. A user submits a question.
4. Dense retrieval and BM25 retrieval run in parallel.
5. Results are merged using Reciprocal Rank Fusion (RRF).
6. Cross-Encoder re-ranks the retrieved chunks.
7. Top-ranked chunks are sent to the LLM.
8. The LLM generates a context-grounded response.

---

## 🎯 Example Use Cases

* Research Paper Question Answering
* Enterprise Knowledge Base Search
* Technical Documentation Assistant
* Internal Document Search
* Academic PDF Analysis

---

## 📚 Key Concepts Demonstrated

* Retrieval-Augmented Generation (RAG)
* Hybrid Search (Dense + Sparse Retrieval)
* Vector Embeddings
* BM25 Ranking
* Reciprocal Rank Fusion (RRF)
* Cross-Encoder Re-ranking
* Prompt Engineering
* LLM Application Development

---

## 👨‍💻 Author

**Jayesh Suthar**

BCA Graduate (Data Science & AI)
Data Science | Data Analyst 

---

## 📜 License

This project is intended for educational, research, and portfolio purposes.
